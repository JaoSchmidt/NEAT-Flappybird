#include "NEAT/Population.h"

#include <pain.h>

#include <utility>
#include <vector>

#include "CoreFiles/LogWrapper.h"
#include "ECS/Components/NativeScript.h"

static constexpr std::string_view materialGameFrame = "GameFrame";

reg::Entity Population::create(pain::Scene &scene, pain::Application &app)
{
  auto [pcs, obstacleMaterial, obstacles] =
      createHelper(scene, app, s_numberOfPlayers);

  reg::Entity game = scene.createEntity("PopulationGame");
  scene.createComponents(game, cmp::Script{});
  const pain::AppInit &config = app.getCurrentConfig();
  const float zoom = app.getCurrentConfig().defaultZoom2d;
  reg::Entity camEntity = pain::Dummy2dCamera::createMovingCamera(
      scene, config.defaultWidth, config.defaultHeight, zoom,
      glm::vec2(0.06f, -0.58f));

  reg::Entity graphRender =
      GraphRender::create(scene, app.getRenderApi(), camEntity);
  // MousePointer::create(scene, app.getRenderApi(), graphRender);
  // reg::Entity graphRender = reg::Entity{-1};
  pain::Scene::emplaceScript<Population>(
      scene.getEntity(), scene, std::move(pcs), obstacleMaterial,
      std::move(obstacles), app, graphRender, camEntity);
  return game;
}

Population::Population(reg::Entity entity, pain::Scene &scene,
                       std::vector<PlayerController *> pcs, pain::Material &om,
                       std::vector<ObstaclesController *> obc,
                       pain::Application &a, reg::Entity graphRender,
                       reg::Entity camEntity)
    : FlappyGame(entity, scene, pcs.at(0), om, std::move(obc), a),
      worldScene(scene), m_rng(2727797253), m_graphRender(graphRender),
      m_camEntity(camEntity)
{
  m_playerControllers = std::move(pcs);
  // every player flies against the same set of obstacles
  for (size_t i = 1; i < m_playerControllers.size(); i++)
    m_playerControllers[i]->m_obstacles = m_playerControllers[0]->m_obstacles;
}

void Population::onCreate()
{
  FlappyGame::onCreate();

  painless::customPanel::addToPanel("Controller", [this]() {
    if (ImGui::Button("Toogle auto time multiplier")) {
      if (m_app.isSimulation())
        m_app.setInfiniteSimulation(false);
      else
        m_app.setInfiniteSimulation(true);
    }
    ImGui::Text("Auto Multiplier is %s", m_app.isSimulation() ? "ON" : "OFF");
    if (ImGui::Button("Toogle NEAT")) {
      m_toggleNEAT = !m_toggleNEAT;
    }
  });

  m_gameScore = 0;

  m_config.m_generation = 0;
  m_config.m_populationSize = 150; // Set the population size
  m_config.m_numInputs = 4;        // Set the number of inputs
  m_config.m_numOutputs = 1;       // Set the number of outputs

  // Non-structural mutation parameters
  m_config.m_initMean = 0.0;             // Set the initial mean
  m_config.m_initStdev = 1.0;            // Set the initial standard deviation
  m_config.m_min = -2.0;                 // Set the minimum value for mutations
  m_config.m_max = 2.0;                  // Set the maximum value for mutations
  m_config.m_mutationRate = 0.8;         // Set the mutation rate
  m_config.m_mutationPower = 0.2;        // Set the mutation power
  m_config.m_replacementRate = 0.05;     // Set the replace rate for links
  m_config.m_biasMutationRate = 0.2;     // Set the bias mutation rate
  m_config.m_biasReplacementRate = 0.05; // Set the replace rate for biases

  // Delta formula parameters
  m_config.m_c1 = 1.0;       // Set c1 parameter for delta
  m_config.m_c2 = 1.0;       // Set c2 parameter for delta
  m_config.m_c3 = 0.4;       // Set c3 parameter for delta
  m_config.dThreshold = 3.0; // Set delta threshold

  // Structural mutations probabilities
  m_config.m_probAddNode = 0.04;  // Set probability of adding a node
  m_config.m_probAddConn = 0.075; // Set probability of adding a connection
  m_config.m_probRmNode = 0.01;   // Set probability of removing a node
  m_config.m_probRmConn = 0.025;  // Set probability of removing a connection

  // NEAT ========================================================== //
  PLOG_I("--- RNG SEED USED = {} -------------------------------------",
         m_rng.seed());
  m_individuals.reserve(m_config.m_populationSize);
  for (int i = 0; i < m_config.m_populationSize; ++i) {
    m_individuals.emplace_back(createMinimalGenome(i), m_config, m_rng, 0);
  }
  m_speciesRepresentatives.emplace(0, m_individuals[0].clone());
  worldScene.getNativeScript<GraphRender>(m_graphRender)
      .generateGraph(worldScene, m_individuals[0].getGenome().m_links,
                     inputInfos(), m_app);

  // PLAYER INPUT ================================================== //
  const int numPlayers = static_cast<int>(m_playerControllers.size());
  m_playerY.resize(numPlayers);
  m_playerVy.resize(numPlayers);
  m_playerAlive.assign(numPlayers, true);
  m_waveBase = 0;
  m_deadThisWave = 0;
  for (int p = 0; p < numPlayers; p++) {
    PlayerController *pc = m_playerControllers[p];
    cmp::Pos2d &ptc = pc->getComponent<cmp::Pos2d>();
    cmp::Mov2d &pmc = pc->getComponent<cmp::Mov2d>();

    m_playerY[p] = &ptc.m_position.y;
    m_playerVy[p] = &pmc.m_velocity.y;
  }

  // PLAYER BOX ================================================== //
  pain::Shader &gameShader =
      m_app.getRenderApi().m_shaderManager.loadShaderFromFile(
          "GameFrame", "resources/shaders/gameFrame.glsl");
  pain::MaterialManager &mm = m_app.getRenderApi().m_materialManager;

  getScene().createComponents(
      getScene().createEntity("PopulationBox"), cmp::Pos2d::create({{0, 0}}),
      cmp::Sprite{
          .layer = pain::RenderLayer::C,
          .m_shape = pain::RectShape{.size = {10.f, 10.f}} //
      },
      cmp::Material::create(
          mm, "PopulationBox",
          {.color = pain::Colors::Brown, .shader = gameShader}) //
  );
}
// ================================================================== //
// ================================================================== //
// GAME RELATED FUNCTIONS
// ================================================================== //
// ================================================================== //

void Population::onUpdate(pain::DeltaTime deltaTime)
{
  const float deltaTimeSec = deltaTime.getSecondsf();
  // spawn obstacles
  m_obstaclesInterval -= m_intervalTime * deltaTimeSec;
  if (m_obstaclesInterval <= 0) {
    m_obstaclesInterval = m_maxInterval;
    const float randAngle =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * M_PI * 2;

    reviveObstacle(m_recentObstacleIndex, randAngle, true);
    m_recentObstacleIndex = (m_recentObstacleIndex + 1) % s_numberOfObstacles;
    reviveObstacle(m_recentObstacleIndex, randAngle, false);
    m_recentObstacleIndex = (m_recentObstacleIndex + 1) % s_numberOfObstacles;
  }

  if (m_gameScore > 300 && !m_soloMode) {
    enterSoloMode();
    return;
  }

  GraphRender *gr = nullptr;
  if (!m_app.isSimulation())
    gr = &worldScene.getNativeScript<GraphRender>(m_graphRender);

  // wave mode: all alive players fly
  for (int playerIdx = 0;
       playerIdx < static_cast<int>(m_playerControllers.size()); playerIdx++) {
    PlayerController *pc = m_playerControllers[playerIdx];
    if (!m_playerAlive[playerIdx])
      continue;

    std::vector<ObstaclesController *> &visible = pc->getVisibleObstacles();

    bool died = false;
    for (ObstaclesController *obs : visible) {
      if (checkPlayerDeath(playerIdx, obs)) {
        died = true;
        break;
      }
    }
    if (died)
      continue;

    ObstaclesController *closestDown = nullptr;
    float closestX = 1.f;
    for (ObstaclesController *obs : visible) {
      if (!obs->isUpsideDown())
        continue;
      float x = obs->getComponent<cmp::Pos2d>().m_position.x;
      if (x < closestX) {
        closestX = x;
        closestDown = obs;
      }
    }
    if (!closestDown)
      continue;

    const glm::vec2 obsPos =
        closestDown->getComponent<cmp::Pos2d>().m_position;

    Individual &individual = m_individuals[m_waveBase + playerIdx];
    if (gr && m_bestIndividualInsideWaveIndex == playerIdx)
      // gt.updateWeights is called
      pc->m_automaticJump = individual.fit(
          {TP_VEC2(obsPos), *m_playerY[playerIdx], *m_playerVy[playerIdx]},
          *gr);
    else
      pc->m_automaticJump = individual.fit(
          {TP_VEC2(obsPos), *m_playerY[playerIdx], *m_playerVy[playerIdx]});
  }
}

bool Population::checkPlayerDeath(int playerIdx, ObstaclesController *obstacle)
{
  if (!obstacle)
    return false;
  PlayerController *pc = m_playerControllers[playerIdx];
  float x = obstacle->getComponent<cmp::Pos2d>().m_position.x;
  if (x < -0.2F && pc->checkIntersection(*obstacle)) {
    afterLosing(playerIdx);
    return true;
  }
  return false;
}

void Population::afterLosing(int playerIdx)
{
  m_individuals[m_waveBase + playerIdx].m_fitness = m_gameScore;
  m_loses++;

  PlayerController *pc = m_playerControllers[playerIdx];
  pc->onDestroy();

  if (m_soloMode) {
    // clear obstacles and reset for a fresh solo attempt
    for (int i = 0; i < s_numberOfObstacles; i++)
      m_playerControllers[0]->m_obstacles[i]->revive(0, 0, false, &m_gameScore);
    m_gameScore = 0;
    m_recentObstacleIndex = 0;
    m_playerAlive[playerIdx] = true;
    return;
  }

  // wave mode: deactivate player and advance when the whole wave is dead
  m_playerAlive[playerIdx] = false;
  m_deadThisWave++;
  const int playersInWave =
      std::min(static_cast<int>(m_playerControllers.size()),
               m_config.m_populationSize - m_waveBase);
  if (m_deadThisWave >= playersInWave)
    nextWave();
}

void Population::nextWave()
{
  // clear all obstacles so leftover ghosts don't kill the next wave
  for (int i = 0; i < s_numberOfObstacles; i++)
    m_playerControllers[0]->m_obstacles[i]->revive(0, 0, false, &m_gameScore);
  m_gameScore = 0;
  m_recentObstacleIndex = 0;

  m_deadThisWave = 0;
  m_waveBase += static_cast<int>(m_playerControllers.size());
  if (m_waveBase >= m_config.m_populationSize) {
    // every individual was evaluated, evolve the population
    if (m_toggleNEAT)
      updateGeneration();
    m_waveBase = 0;
  }

  m_bestIndividualInsideWaveIndex = 0;
  Individual &bestIndividualInsideWave = m_individuals[m_waveBase];
  const int numPlayers = static_cast<int>(m_playerControllers.size());
  for (int p = 0; p < numPlayers; p++) {
    const bool active = (m_waveBase + p) < m_config.m_populationSize;
    m_playerAlive[p] = active;
    PlayerController *pc = m_playerControllers[p];
    pc->resetPosition();
    Individual &individual = m_individuals[m_waveBase + p];

    if (bestIndividualInsideWave.m_fitness < individual.m_fitness) {
      m_bestIndividualInsideWaveIndex = p;
    }
  }
}

void Population::enterSoloMode()
{
  const int numPlayers = static_cast<int>(m_playerControllers.size());

  // find the alive player to identify the champion individual
  int aliveSlot = 0;
  for (int p = 0; p < numPlayers; p++) {
    if (m_playerAlive[p]) {
      aliveSlot = p;
      break;
    }
  }
  m_soloIdx = m_waveBase + aliveSlot;
  m_soloMode = true;

  // clear obstacles and reset the score
  for (int i = 0; i < s_numberOfObstacles; i++)
    m_playerControllers[0]->m_obstacles[i]->revive(0, 0, false, &m_gameScore);
  m_gameScore = 0;
  m_recentObstacleIndex = 0;

  // downsize to a single solo player
  m_playerControllers.resize(1);
  m_playerY.resize(1);
  m_playerVy.resize(1);
  m_playerAlive.assign(1, true);

  // reset the solo player
  m_playerControllers[0]->resetPosition();

  // stop the NEAT run, restore normal rendering
  m_toggleNEAT = false;
  m_app.setTimeMultiplier(1.0);
  m_app.setRendereing(true);
}

// ================================================================== //
// ================================================================== //
// NEAT RELATED FUNCTIONS
// ================================================================== //
// ================================================================== //

void Population::speciateFitness()
{
  LOG_I("RepSize = {}, IndSize = {}", m_speciesRepresentatives.size(),
        m_individuals.size());
  std::vector<int> speciesCount(m_speciesRepresentatives.size(), 0);
  m_speciesInfo.clear();
  m_speciesInfo.reserve(m_speciesRepresentatives.size());

  // Count individuals in each species
  for (const auto &individual : m_individuals) {
    speciesCount[individual.m_speciesID]++;
  }
  for (unsigned i = 0; i < speciesCount.size(); i++) {
    m_speciesInfo.emplace(i, SpeciesFit(speciesCount[i], 0.0));
  }

  // Calculate shared fitness for each individual
  for (Individual &individual : m_individuals) {
    int speciesSize = speciesCount[individual.m_speciesID];
    individual.m_fitness /= speciesSize; // Apply fitness sharing
    m_speciesInfo[individual.m_speciesID].avereageFitness +=
        individual.m_fitness;
  }

  // calculate stacked area graphic
  for (int i = 0; i < static_cast<int>(m_speciesInfo.size()); i++) {
    if (m_speciesInfo[i].count != 0)
      m_speciesInfo[i].avereageFitness /= m_speciesInfo[i].count;
  }
}

void Population::classifyAllSpecies()
{
  for (auto &individual : m_individuals) {
    bool foundSpecies = false;
    if (individual.m_fitness > m_bestIndividual->m_fitness)
      m_bestIndividual = &individual;

    // Attempt to classify into an existing species
    for (const auto &[speciesID, representative] : m_speciesRepresentatives) {
      double delta = individual.calculateDelta(representative);

      // Check if individual belongs to this species
      if (delta < m_config.dThreshold) {
        individual.m_speciesID = speciesID;
        foundSpecies = true;

        // Update species representative if individual has better fitness
        if (individual.m_fitness >= representative.m_fitness) {
          m_speciesRepresentatives.at(speciesID) = individual.clone();
        }
        break;
      }
    }

    // If no compatible species was found, create a new one
    if (!foundSpecies) {
      int nextSpeciesID = static_cast<int>(m_speciesRepresentatives.size());
      individual.m_speciesID = nextSpeciesID;
      m_speciesRepresentatives.emplace(nextSpeciesID, individual.clone());
    }
  }
  GraphRender &gr = worldScene.getNativeScript<GraphRender>(m_graphRender);
  gr.generateGraph(worldScene, m_bestIndividual->getGenome().m_links,
                   inputInfos(), m_app);
}

std::vector<Individual>
Population::tournamentSelection(int numToSurvive, int tournamentSize) const
{
  std::vector<Individual> selectedIndividuals;
  std::random_device rd;
  std::mt19937 gen(rd());

  selectedIndividuals.reserve(numToSurvive);
  for (int i = 0; i < numToSurvive; ++i) {
    std::unique_ptr<Individual> bestIndividual = nullptr;

    // Create a tournament of randomly selected individuals
    for (int j = 0; j < tournamentSize; ++j) {
      int randIndex = gen() % m_individuals.size();
      // clang-format off
      if (!bestIndividual || m_individuals[randIndex].m_fitness > bestIndividual->m_fitness)
        bestIndividual = std::make_unique<Individual>(m_individuals[randIndex].clone());
      // clang-format on
    }

    // Add a copy of the fittest individual from to the selection
    selectedIndividuals.push_back(std::move(*bestIndividual));
  }

  return selectedIndividuals;
}

void Population::offspringAndMutate(std::vector<Individual> selectedIndividuals)
{
  std::vector<Individual> offspring;
  offspring.reserve(m_config.m_populationSize - selectedIndividuals.size());

  // Generate offspring until we reach the desired population size
  while (selectedIndividuals.size() + offspring.size() <
         (unsigned)m_config.m_populationSize) {
    // Select two random parents from the selected individuals
    const Individual &parent1 = selectedIndividuals[m_rng.uniform<int>(
        0, selectedIndividuals.size() - 1)];
    const Individual &parent2 = selectedIndividuals[m_rng.uniform<int>(
        0, selectedIndividuals.size() - 1)];

    // Create an offspring through crossover
    Individual child = parent1.crossover(parent2, m_populationInnov);

    // Apply mutations to the offspring
    child.mutateAddNeuron();
    child.mutateAddLink();
    child.mutateRemoveNeuron();
    child.mutateRemoveLink();

    child.nonStructuralMutate();

    // Add the offspring to the temporary vector
    offspring.push_back(std::move(child));
  }

  m_individuals.clear();
  m_individuals.reserve(m_config.m_populationSize);

  // Move selected individuals and offspring
  m_individuals = std::move(selectedIndividuals);
  m_individuals.insert(m_individuals.end(),
                       std::make_move_iterator(offspring.begin()),
                       std::make_move_iterator(offspring.end()));
}

void Population::updateGeneration()
{
  // speciation
  classifyAllSpecies();
  speciateFitness();

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(8);
  for (auto &species : m_speciesInfo) {
    if (species.second.count != 0)
      oss << "(" << species.first << ", " << species.second.count << ": "
          << species.second.avereageFitness << "), ";
  }
  LOG_T("{} (species, count: average): [{}]", ++m_config.m_generation - 1,
        oss.str());

  // selection
  std::vector<Individual> selection =
      tournamentSelection(m_individuals.size() / 3, m_individuals.size() * 0.2);
  // mutation and recombination/crossover
  offspringAndMutate(std::move(selection));
}

std::vector<InputInfo> &Population::inputInfos()
{
  static std::vector<InputInfo> inputInfos = {{"obstacleX", 2.f, -1.f},
                                              {"obstacleY", 1.45f, -1.95},
                                              {"playerY", 1.f, -1.f},
                                              {"playerVy", 1.f, -1.f}};
  return inputInfos;
}

Genome Population::createMinimalGenome(int individualIndex)
{
  std::vector<NodeGene> neurons = {};

  // inputs
  neurons.reserve(5);
  neurons.emplace_back(-1, m_rng.gaussian<double>()); // obstacleX
  neurons.emplace_back(-2, m_rng.gaussian<double>()); // obstacleY
  neurons.emplace_back(-3, m_rng.gaussian<double>()); // playerY
  neurons.emplace_back(-4, m_rng.gaussian<double>()); // playerVy
  // outputs
  int outputId = -static_cast<int>(neurons.size() + 1);
  neurons.emplace_back(outputId, 0.0);

  // links and innovations
  std::vector<ConnectionGene> links = {};
  for (int id = 1; id < 5; id++) {
    links.emplace_back(-id, outputId, m_rng.gaussian<double>(), true, id);
    if (individualIndex == 0) // excpected to work once
      m_populationInnov.emplace_back(-id, outputId, id - 1);
  }

  return Genome(std::move(neurons), std::move(links), m_populationInnov);
}

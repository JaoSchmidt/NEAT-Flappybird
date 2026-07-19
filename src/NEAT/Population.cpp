#include "NEAT/Population.h"
#include "CoreFiles/LogWrapper.h"
#include "ECS/Components/NativeScript.h"
#include <pain.h>
#include <utility>
#include <vector>

reg::Entity Population::create(pain::Scene &scene, pain::Application &app) {

  auto [pc, obstacleMaterial, obstacles] = createHelper(scene, app);

  reg::Entity game = scene.createEntity();
  scene.createComponents(game, pain::NativeScriptComponent{});
  const pain::AppInit &config = app.getCurrentConfig();
  const float zoom = app.getCurrentConfig().defaultZoom2d;
  reg::Entity camEntity = pain::Dummy2dCamera::createMovingCamera(
      scene, config.defaultWidth, config.defaultHeight, zoom);

  reg::Entity graphRender =
      GraphRender::create(scene, app.getRenderApi(), camEntity);
  // MousePointer::create(scene, app.getRenderApi(), graphRender);
  // reg::Entity graphRender = reg::Entity{-1};
  pain::Scene::emplaceScript<Population>(scene.getEntity(), scene, pc,
                                         obstacleMaterial, std::move(obstacles),
                                         app, graphRender);
  return game;
}

Population::Population(reg::Entity entity, pain::Scene &scene,
                       PlayerController *pc, pain::Material &om,
                       std::vector<ObstaclesController *> obc,
                       pain::Application &a, reg::Entity graphRender)
    : FlappyGame(entity, scene, pc, om, std::move(obc), a), worldScene(scene),
      m_graphRender(graphRender) {};

void Population::onCreate() {
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
    ImGui::Text("is NEAT running? %s", m_rendering ? "ON" : "OFF");
  });

  m_points = 0;

  m_config.m_generation = 0;
  m_config.m_populationSize = 150; // Set the population size
  m_config.m_numInputs = 5;        // Set the number of inputs
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
  m_individuals.reserve(m_config.m_populationSize);
  for (int i = 0; i < m_config.m_populationSize; ++i) {
    m_individuals.emplace_back(createMinimalGenome(i), m_config, m_rng, 0);
  }
  m_speciesRepresentatives.emplace(0, m_individuals[0].clone());
  m_currentObsIndex = m_index;
  worldScene.getNativeScript<GraphRender>(m_graphRender)
      .generateGraph(worldScene, m_individuals[0].getGenome().m_links,
                     m_individuals[0].getGenome().m_neurons, m_app);

  // PLAYER INPUT ================================================== //
  pain::Transform2dComponent &ptc =
      m_playerController->getComponent<pain::Transform2dComponent>();
  pain::Movement2dComponent &pmc =
      m_playerController->getComponent<pain::Movement2dComponent>();
  pain::RotationComponent &prc =
      m_playerController->getComponent<pain::RotationComponent>();

  m_playerY = &ptc.m_position.y;
  m_playerVy = &pmc.m_velocity.y;
  m_playerRot = &prc.m_rotationRadians;

  // PLAYER BOX ================================================== //
  pain::Shader &s = m_app.getRenderApi().m_shaderManager.getDefaultShader(
      pain::DefaultShader::Texture);
  pain::Material &m = m_app.getRenderApi().m_materialManager.createMaterial(
      "Boxes", {.color = pain::Colors::Brown, .shader = s});

  reg::Entity box = getScene().createEntity();
  getScene().createComponents(
      box,
      pain::Transform2dComponent::create({{0, PlayerController::MAX_HEIGHT}}),
      pain::SpriteComponent{.layer = pain::RenderLayer::G},
      pain::MaterialComponent::create(m) //
  );
}
// ================================================================== //
// ================================================================== //
// GAME RELATED FUNCTIONS
// ================================================================== //
// ================================================================== //

// ** get index of the closest obstacle to the left of the player
int Population::getClosestObstacle(float playerPos) {
  for (unsigned i = 0; i < m_obstacles.size(); i++) {
    int index = (m_currentObsIndex + i) % m_obstacles.size();
    float &obstaclePosX = m_obstacles[index]
                              ->getComponent<pain::Transform2dComponent>()
                              .m_position.x;
    if (obstaclePosX > playerPos && obstaclePosX < 0.08) {
      return index;
    }
  }
  return m_currentObsIndex;
}

void Population::onUpdate(pain::DeltaTime deltaTime) {

  // spawn obstacles
  m_obstaclesInterval -= m_intervalTime * deltaTime.getSeconds();
  if (m_obstaclesInterval <= 0) {
    m_obstaclesInterval = m_maxInterval;
    const float randAngle =
        static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * M_PI * 2;

    reviveObstacle(m_index, randAngle, true);
    m_index = (m_index + 1) % s_numberOfObstacles;
    reviveObstacle(m_index, randAngle, false);
    m_index = (m_index + 1) % s_numberOfObstacles;
  }
  // check if m_points changed
  if (m_points > m_pointsChecker) {
    m_pointsChecker = m_points;
    m_currentObsIndex = getClosestObstacle(DEFAULTXPOS);
  }

  // check collision and losing state
  for (char i = 0; i < s_numberOfObstacles; i++) {
    ObstaclesController &obstacle = *m_obstacles.at(i);
    const auto &tc = obstacle.getComponent<pain::Transform2dComponent>();
    // no extra life for now
    if (tc.m_position.x < -0.2F && checkIntersection(obstacle)) {
      afterLosing();
      return;
    }
    // Assuming 300 is
    if (m_points > 300) {
      m_app.setTimeMultiplier(1.);
      m_app.setRendereing(true);
    }
  }

  const glm::vec2 obsPos = m_obstacles[m_currentObsIndex]
                               ->getComponent<pain::Transform2dComponent>()
                               .m_position;

  // if (obsPos.x < DEFAULTXPOS)
  //   m_currentObsIndex = getClosestObstacle(DEFAULTXPOS);

  // INPUT VARIABLES, including player and obstacles
  if (m_app.isSimulation()) {
    m_playerController->m_automaticJump = m_individuals[m_currentIndIndex].fit(
        {TP_VEC2(obsPos), *m_playerY, *m_playerVy, *m_playerRot});
  } else {
    GraphRender &gr = worldScene.getNativeScript<GraphRender>(m_graphRender);
    m_playerController->m_automaticJump = m_individuals[m_currentIndIndex].fit(
        {TP_VEC2(obsPos), *m_playerY, *m_playerVy, *m_playerRot}, gr);
  }
}

void Population::afterLosing() {
  // update fitness
  m_individuals[m_currentIndIndex].m_fitness = m_points;
  // LOG_I("Individual {}, Pontuation = {}", m_currentIndIndex, m_points);
  if (m_toggleNEAT) {
    if (m_currentIndIndex == m_config.m_populationSize - 1)
      updateGeneration();
    m_currentIndIndex = (m_currentIndIndex + 1) % m_config.m_populationSize;
  }
  m_loses++;
  FlappyGame::afterLosing();
}

// ================================================================== //
// ================================================================== //
// NEAT RELATED FUNCTIONS
// ================================================================== //
// ================================================================== //

void Population::speciateFitness() {
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

void Population::classifyAllSpecies() {

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
                   m_bestIndividual->getGenome().m_neurons, m_app);
}

std::vector<Individual>
Population::tournamentSelection(int numToSurvive, int tournamentSize) const {
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

void Population::offspringAndMutate(
    std::vector<Individual> selectedIndividuals) {
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

void Population::updateGeneration() {
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

Genome Population::createMinimalGenome(int individualIndex) {
  std::vector<NodeGene> neurons = {};
  int outputId = -6;

  // inputs
  neurons.reserve(6);
  neurons.emplace_back(-1, m_rng.gaussian<double>()); // obstacleX
  neurons.emplace_back(-2, m_rng.gaussian<double>()); // obstacleY
  neurons.emplace_back(-3, m_rng.gaussian<double>()); // playerY
  neurons.emplace_back(-4, m_rng.gaussian<double>()); // playerVy
  neurons.emplace_back(-5, m_rng.gaussian<double>()); // playerRot
  // outputs
  neurons.emplace_back(outputId, 0.0);

  // links and innovations
  std::vector<ConnectionGene> links = {};
  for (int id = 1; id < 6; id++) {
    links.emplace_back(-id, outputId, m_rng.gaussian<double>(), true, id);
    if (individualIndex == 0) // excpected to work once
      m_populationInnov.emplace_back(-id, outputId, id - 1);
  }

  return Genome(std::move(neurons), std::move(links), m_populationInnov);
}

#include "NEAT/Population.h"
#include "ECS/Components/Movement.h"
#include "SDL_events.h"
#include <algorithm>
#include <iostream>
#include <utility>

void Population::onRender(double currentTime) {}
void Population::onEvent(const SDL_Event &event) {}

void Population::onCreate()
{
  const int w = 1024;
  const int h = 768;
  std::shared_ptr<pain::OrthoCameraEntity> camera =
      std::make_shared<pain::OrthoCameraEntity>(this, (float)w / h, 1.0f);
  pain::Renderer2d::init(camera);
  camera->addComponent<pain::NativeScriptComponent>()
      .bind<pain::OrthoCameraController>();

  auto texture =
      std::make_shared<pain::Texture>("resources/textures/Player.png");
  m_pplayer = std::make_unique<Player>(this, texture);
  pain::NativeScriptComponent &pNSC =
      m_pplayer->addComponent<pain::NativeScriptComponent>();
  pNSC.bind<PlayerController>();
  initializeScripts(pNSC, *m_pplayer);

  m_obstacles.reserve(m_numberOfObstacles);
  for (char i = 0; i < m_numberOfObstacles; i++) {
    m_obstacles.emplace_back(this);
    pain::NativeScriptComponent &oNSC =
        m_obstacles[i].addComponent<pain::NativeScriptComponent>();
    oNSC.bind<ObstaclesController>();
    initializeScripts(oNSC, m_obstacles[i]);
  };

  pain::Application::Get().addImGuiInstance((ImGuiInstance *)this);
  // pain::Application::Get().disableRendering();
  *(pain::Application::Get().getTimeMultiplier()) = 30.0;

  // ============================================================ //
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

  m_individuals.reserve(m_config.m_populationSize);
  for (int i = 0; i < m_config.m_populationSize; ++i) {
    m_individuals.emplace_back(createMinimalGenome(i), m_config, m_rng, 0);
  }
  m_speciesRepresentatives.emplace(0, m_individuals[0].clone());

  m_currentObsIndex = m_index;
  pain::TransformComponent &tc =
      m_obstacles[m_currentObsIndex].getComponent<pain::TransformComponent>();
  m_obstacleX = &tc.m_position.x;
  m_obstacleY = &tc.m_position.y;

  pain::TransformComponent &ptc =
      m_pplayer->getComponent<pain::TransformComponent>();
  pain::MovementComponent &pmc =
      m_pplayer->getComponent<pain::MovementComponent>();
  pain::RotationComponent &prc =
      m_pplayer->getComponent<pain::RotationComponent>();

  m_playerY = &ptc.m_position.y;
  m_playerVy = &pmc.m_velocityDir.y;
  m_playerRot = &prc.m_rotationAngle;
}

// ** get index of the closest obstacle to the left of the player
int Population::getClosestObstacle(float playerPos)
{
  for (unsigned i = 0; i < m_obstacles.size(); i++) {
    int index = (m_currentObsIndex + i) % m_obstacles.size();
    float &obstaclePosX = m_obstacles[index]
                              .getComponent<pain::TransformComponent>()
                              .m_position.x;
    if (obstaclePosX > playerPos && obstaclePosX < 0.2) {
      return index;
    }
  }
  return m_currentObsIndex;
}

void Population::onUpdate(double deltaTime)
{
  // spawn obstacles
  m_obstaclesInterval -= m_intervalTime * deltaTime;
  if (m_obstaclesInterval <= 0) {
    m_obstaclesInterval = m_maxInterval;
    const float randAngle = ((float)rand() / RAND_MAX) * M_PI * 2;

    m_index = (m_index + 1) % m_numberOfObstacles;
    reviveObstacle(m_index, randAngle, true);
    m_index = (m_index + 1) % m_numberOfObstacles;
    reviveObstacle(m_index, randAngle, false);
  }
  // check if m_points changed
  if (m_points > m_pointsChecker) {
    m_pointsChecker = m_points;
    m_currentObsIndex = getClosestObstacle(DEFAULTXPOS);
    pain::TransformComponent &tc =
        m_obstacles[m_currentObsIndex].getComponent<pain::TransformComponent>();
    m_obstacleX = &tc.m_position.x;
    m_obstacleY = &tc.m_position.y;
  }

  // check collision and losing state
  for (char i = 0; i < m_numberOfObstacles; i++) {
    Obstacles &obstacle = m_obstacles.at(i);
    const auto &tc = obstacle.getComponent<pain::TransformComponent>();
    // no extra life for now
    if (tc.m_position.x < -0.2f && checkIntersection(*m_pplayer, obstacle, i)) {
      afterLosing();
      return;
    }
    if (m_points > 300)
      LOG_I("Individual {} of species {} reached 300 points", m_currentIndIndex,
            m_individuals[m_currentIndIndex].m_speciesID);
  }

  // calculate space pressed probability
  ASSERT(m_obstacleX != nullptr, "m_obstacleX is null");
  ASSERT(m_obstacleY != nullptr, "m_obstacleY is null");
  ASSERT(m_playerY != nullptr, "m_playerY is null");
  ASSERT(m_playerVy != nullptr, "m_playerVy is null");
  ASSERT(m_playerRot != nullptr, "m_playerRot is null");
  if (m_individuals[m_currentIndIndex].fit({*m_obstacleX, *m_obstacleY,
                                            *m_playerY, *m_playerVy,
                                            *m_playerRot})) {
    // SDL_PushEvent not working
    ((PlayerController *)m_pplayer->getComponent<pain::NativeScriptComponent>()
         .instance)
        ->m_automaticJump = true;
  } else {
    ((PlayerController *)m_pplayer->getComponent<pain::NativeScriptComponent>()
         .instance)
        ->m_automaticJump = false;
  }
}

void Population::afterLosing()
{
  // update fitness
  m_individuals[m_currentIndIndex].m_fitness = m_points;
  // LOG_I("Individual {}, Pontuation = {}", m_currentIndIndex, m_points);
  if (m_toggleNEAT) {
    if (m_currentIndIndex == m_config.m_populationSize - 1)
      updateGeneration();
    m_currentIndIndex = (m_currentIndIndex + 1) % m_config.m_populationSize;
  }
  m_loses++;
  m_points = 0;
  // reset Player position
  ((PlayerController *)m_pplayer->getComponent<pain::NativeScriptComponent>()
       .instance)
      ->resetPosition();
  // clear obstacles
  for (char i = 0; i < m_numberOfObstacles; i++) {
    Obstacles &obstacle = m_obstacles.at(i);
    auto *inst = (ObstaclesController *)obstacle
                     .getComponent<pain::NativeScriptComponent>()
                     .instance;
    inst->revive(0, 0, false, &m_points);
  }
}

void Population::speciateFitness()
{
  std::vector<int> speciesCount(m_speciesRepresentatives.size(), 0);

  // Count individuals in each species
  for (const auto &individual : m_individuals) {
    speciesCount[individual.m_speciesID]++;
  }

  // Calculate shared fitness for each individual
  for (auto &individual : m_individuals) {
    int speciesSize = speciesCount[individual.m_speciesID];
    individual.m_fitness /= speciesSize; // Apply fitness sharing
  }
}

void Population::classifyAllSpecies()
{
  std::map<int, Individual>
      newSpeciesRepresentatives; // Temporary storage for updated
                                 // representatives

  for (auto &individual : m_individuals) {
    bool foundSpecies = false;

    // Attempt to classify into an existing species
    for (const auto &[speciesID, representative] : m_speciesRepresentatives) {
      double delta = individual.calculateDelta(representative);

      // Check if individual belongs to this species
      if (delta < m_config.dThreshold) {
        individual.m_speciesID = speciesID;
        foundSpecies = true;

        // Update species representative if individual has better fitness
        if (individual.m_fitness >= representative.m_fitness) {
          newSpeciesRepresentatives.emplace(speciesID, individual.clone());
        }
        break;
      }
    }

    // If no compatible species was found, create a new one
    if (!foundSpecies) {
      int nextSpeciesID = m_speciesRepresentatives.size();
      individual.m_speciesID = nextSpeciesID;
      newSpeciesRepresentatives.emplace(nextSpeciesID, individual.clone());
    }
  }

  // Update the list of species representatives with the new generation's
  // representatives
  m_speciesRepresentatives = std::move(newSpeciesRepresentatives);
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
  oss << std::fixed << std::setprecision(2);
  for (Individual &ind : m_individuals) {
    oss << ind.m_fitness << ", ";
  }
  LOG_T("{}: [{}]", ++m_config.m_generation - 1, oss.str());

  // selection
  std::vector<Individual> selection =
      tournamentSelection(m_individuals.size() / 2, m_individuals.size() * 0.1);
  // mutation and recombination/crossover
  offspringAndMutate(std::move(selection));
}

Genome Population::createMinimalGenome(int individualIndex)
{
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
  // layers
  std::vector<int> layers = {0, 1};

  return Genome(std::move(neurons), std::move(links), std::move(layers),
                m_populationInnov);
}

void Population::reviveObstacle(int index, float randomAngle, bool upsideDown)
{
  const float height = upsideDown
                           ? sin(randomAngle) * 0.7 + 0.75f + m_obstaclesSpacing
                           : sin(randomAngle) * 0.7 - 1.25f;
  ((ObstaclesController *)m_obstacles.at(index)
       .getComponent<pain::NativeScriptComponent>()
       .instance)
      ->revive(m_defaultObstacleSpeed, height, upsideDown, &m_points);
}

bool Population::checkIntersection(const Player &player,
                                   const Obstacles &obstacle, int index)
{
  auto &ptc = player.getComponent<pain::TransformComponent>();
  auto &prc = player.getComponent<pain::RotationComponent>();
  auto &psc = player.getComponent<pain::SpriteComponent>();
  auto &otc = obstacle.getComponent<pain::TransformComponent>();
  auto &otgc = obstacle.getComponent<pain::TrianguleComponent>();

  // get quad vertices
  constexpr glm::vec4 quadVertexPositions[4] = {
      glm::vec4(-0.5f, -0.5f, 0.f, 1.f),
      glm::vec4(0.5f, -0.5f, 0.f, 1.f),
      glm::vec4(0.5f, 0.5f, 0.f, 1.f),
      glm::vec4(-0.5f, 0.5f, 0.f, 1.f),
  };
  const glm::mat4 transform = pain::Renderer2d::getTransform(
      ptc.m_position, psc.m_size, prc.m_rotationAngle);

  std::array<glm::vec2, 4> qVertices = {
      transform * quadVertexPositions[0],
      transform * quadVertexPositions[1],
      transform * quadVertexPositions[2],
      transform * quadVertexPositions[3],
  };

  // triangle
  constexpr glm::vec4 triVertexPositions[3] = {
      glm::vec4(0.0f, 0.5f, 0.f, 1.f),
      glm::vec4(0.5f, -0.5f, 0.f, 1.f),
      glm::vec4(-0.5f, -0.5f, 0.f, 1.f),
  };
  const glm::mat4 transformTri =
      pain::Renderer2d::getTransform(otc.m_position, otgc.m_height);
  const std::array<glm::vec2, 3> tVertices = {
      transformTri * triVertexPositions[0], //
      transformTri * triVertexPositions[1], //
      transformTri * triVertexPositions[2], //
  };

  std::vector<glm::vec2> axes;
  for (size_t i = 0; i < 4; i++) {
    glm::vec2 edge = qVertices[(i + 1) % 4] - qVertices[i];
    glm::vec2 axis(-edge.y, edge.x); // Perpendicular to the edge
    axis = glm::normalize(axis);
    axes.push_back(axis);
  }
  for (size_t i = 0; i < 3; i++) {
    glm::vec2 edge = tVertices[(i + 1) % 3] - tVertices[i];
    glm::vec2 axis(-edge.y, edge.x);
    axis = glm::normalize(axis);
    axes.push_back(axis);
  }
  // Perform SAT check on all axes
  for (const glm::vec2 &axis : axes) {
    auto boundA = projection(qVertices, axis);
    auto boundB = projection(tVertices, axis);

    // Check for overlap
    if (boundA.y < boundB.x || boundB.y < boundA.x) {
      return false; // No collision
    }
  }

  return true;
}

template <std::size_t T>
glm::vec2 Population::projection(const std::array<glm::vec2, T> &shape,
                                 const glm::vec2 &axis)
{
  float min = glm::dot(shape[0], axis);
  float max = min;
  for (size_t i = 1; i < shape.size(); i++) {
    float projection = glm::dot(shape[i], axis);
    min = std::min(min, projection);
    max = std::max(max, projection);
  }
  return {min, max};
}

const void Population::onImGuiUpdate()
{
  ImGui::Begin("Player Controller");
  ImGui::Text("Parameters Settings");
  ImGui::InputInt("Number of Obstacles", &m_numberOfObstacles);
  ImGui::InputFloat("Obstacles Spacing", &m_obstaclesSpacing, 0.01f, 1.0f,
                    "%.3f");
  ImGui::InputFloat("Max Interval", &m_maxInterval, 0.1f, 1.0f, "%.3f");
  ImGui::InputFloat("Interval Time", &m_intervalTime, 0.1f, 1.0f, "%.3f");
  ImGui::InputFloat("Obstacle Speed", &m_defaultObstacleSpeed, 0.01f, 1.0f,
                    "%.3f");
  ImGui::InputFloat("Color Interval", &m_colorInterval, 0.1f, 1.0f, "%.3f");
  ImGui::InputFloat("Height Interval", &m_heightInterval, 0.1f, 1.0f, "%.3f");
  ImGui::SeparatorText("Info");
  ImGui::Text("Obstacle Spawn counter: %.2f seconds", m_obstaclesInterval);
  ImGui::Text("Last Obstacle index: %.2d", m_index);
  ImGui::Text("Points: %.4d", m_points);
  ImGui::Text("Loses: %.4d", m_loses);
  ImGui::Text("TPS: %.1f", pain::Application::Get().getCurrentTPS());
  ImGui::Text("Current closest Index %d", m_currentObsIndex);
  ImGui::InputDouble("Time Multiplier",
                     pain::Application::Get().getTimeMultiplier(), 0.1f, 1.0f,
                     "%.3f");
  if (ImGui::Button("Toogle Rendering")) {
    m_rendering = !m_rendering;
    if (m_rendering)
      pain::Application::Get().disableRendering();
    else
      pain::Application::Get().enableRendering();
  }
  ImGui::Text("Rendering is %s", m_rendering ? "ON" : "OFF");
  if (ImGui::Button("Toogle NEAT")) {
    m_toggleNEAT = !m_toggleNEAT;
  }
  ImGui::Text("is NEAT running? %s", m_rendering ? "ON" : "OFF");
  ImGui::End();
}

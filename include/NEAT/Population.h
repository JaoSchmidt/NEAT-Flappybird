#pragma once
#include "FlappyGame.h"
#include "NEAT/GraphRendering/GraphRender.h"
#include "NEAT/Individuals.h"
#include "NEAT/NN.h"
#include <vector>

struct SpeciesFit {
  int count = 0;
  double avereageFitness = 0.0;
};

class Population : public FlappyGame {

public:
  reg::Entity static create(pain::Scene &scene, pain::Application &app);
  void onCreate();
  void onUpdate(pain::DeltaTime deltaTime);

  Population(reg::Entity entity, pain::Scene &scene, PlayerController *pc,
             pain::Material &om, std::vector<ObstaclesController *> obc,
             pain::Application &a, reg::Entity graphRender);

  NONCOPYABLE(Population)
  NONMOVABLE(Population)
  ~Population() = default;

protected:
  pain::Scene &worldScene; // To mess with time multipliers
  // forced delta time equal 1/60
  static constexpr double m_deltaTime = static_cast<double>(1) / 60;
  bool m_rendering = true;
  bool m_toggleNEAT = true;

  NeatConfig m_config = {};
  pain::RNG m_rng;
  std::vector<InnovationStatic> m_populationInnov;
  std::vector<Individual> m_individuals;
  std::unordered_map<int, Individual> m_speciesRepresentatives;
  std::unordered_map<int, SpeciesFit> m_speciesInfo;
  // population stuff
  Genome createMinimalGenome(int individualIndex);
  void updateGeneration();   // speciate + select + combine + mutate
  void classifyAllSpecies(); // attempt to separate into species
  void speciateFitness();    // when applying the "shared" function
  std::vector<Individual> tournamentSelection(int numToSelect,
                                              int tournamentSize) const;
  void offspringAndMutate(std::vector<Individual> selection);

  // inputs from player
  float *m_playerY = nullptr;
  float *m_playerVy = nullptr;
  float *m_playerRot = nullptr;

  int m_pointsChecker = 0;   // detect inputs
  int m_currentObsIndex = 0; // current obstacle index
  int m_currentIndIndex = 0;
  int getClosestObstacle(float playerPosX);
  Individual *m_bestIndividual = nullptr;
  reg::Entity m_graphRender;

  void afterLosing();

  reg::Entity m_frame;
};

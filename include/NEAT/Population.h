#pragma once
#include <vector>

#include "FlappyGame.h"
#include "NEAT/GraphRendering/GraphRender.h"
#include "NEAT/Individuals.h"
#include "NEAT/NN.h"

struct SpeciesFit {
  int count = 0;
  double avereageFitness = 0.0;
};

class Population : public FlappyGame
{
public:
  reg::Entity static create(pain::Scene &scene, pain::Application &app);
  void onCreate();
  void onUpdate(pain::DeltaTime deltaTime);

  Population(reg::Entity entity, pain::Scene &scene,
             std::vector<PlayerController *> pcs, pain::Material &om,
             std::vector<ObstaclesController *> obc, pain::Application &a,
             reg::Entity graphRender, reg::Entity camEntity);

  NONCOPYABLE(Population)
  NONMOVABLE(Population)
  ~Population() = default;

  static std::vector<InputInfo> &inputInfos();

protected:
  pain::Scene &worldScene; // To mess with time multipliers
  // forced delta time equal 1/60
  static constexpr double m_deltaTime = static_cast<double>(1) / 60;
  static constexpr int s_numberOfPlayers = 1;
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

  // players running in parallel, one individual assigned to each
  std::vector<PlayerController *> m_playerControllers;
  // inputs from player
  std::vector<float *> m_playerY;
  std::vector<float *> m_playerVy;

  int m_waveBase = 0; // first individual index of the current wave
  int m_deadThisWave = 0;
  std::vector<bool> m_playerAlive;

  // solo mode: champion runs alone after hitting the point threshold
  bool m_soloMode = false;
  int m_soloIdx = 0;      // index into m_individuals for the champion
  int m_championSlot = 0; // player slot the champion occupies

  int m_bestIndividualInsideWaveIndex = 0;
  Individual *m_bestIndividual = nullptr;
  reg::Entity m_graphRender;
  reg::Entity m_camEntity;

  void afterLosing(int playerIdx);
  void nextWave();
  void enterSoloMode();
  bool checkPlayerDeath(int playerIdx, ObstaclesController *obstacle);

  reg::Entity m_frame;
};

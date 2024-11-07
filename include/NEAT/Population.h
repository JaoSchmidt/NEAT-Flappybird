#pragma once
#include "NEAT/Individuals.h"
#include "NEAT/NN.h"
#include "Obstacles.h"
#include "Player.h"
#include <vector>

class Population : public pain::Scene, public pain::ImGuiInstance
{

public:
  void onCreate();
  void onRender(double currenTime);
  void onUpdate(double deltaTime);
  void onEvent(const SDL_Event &event);
  const void onImGuiUpdate();

private:
  // forced delta time equal 1/60
  static constexpr double m_deltaTime = static_cast<double>(1) / 60;
  bool m_rendering = true;
  bool m_toggleNEAT = true;

  NeatConfig m_config;
  pain::RNG m_rng;
  std::vector<InnovationStatic> m_populationInnov;
  std::vector<Individual> m_individuals;
  std::map<int, Individual> m_speciesRepresentatives;
  // population stuff
  Genome createMinimalGenome(int individualIndex);
  void updateGeneration();   // speciate + select + combine + mutate
  void classifyAllSpecies(); // attempt to separate into species
  void speciateFitness();    // when applying the "shared" function
  std::vector<Individual> tournamentSelection(int numToSelect,
                                              int tournamentSize) const;
  void offspringAndMutate(std::vector<Individual> selection);

  // inputs
  float *m_obstacleX;
  float *m_obstacleY;
  float *m_playerY;
  float *m_playerVy;
  float *m_playerRot;

  int m_pointsChecker = 0;   // detect inputs
  int m_currentObsIndex = 0; // current obstacle index
  int m_currentIndIndex = 0;
  int getClosestObstacle(float playerPosX);

  // Game/Train Options
  int m_numberOfObstacles = 20;
  float m_obstaclesSpacing = 0.35f;
  float m_obstaclesInterval = 1.6;
  float m_intervalTime = 0.6;
  float m_maxInterval = 1.6;
  float m_defaultObstacleSpeed = -0.32;
  float m_colorInterval = 20.;  // color waves
  float m_heightInterval = 20.; // height waves

  float m_waveColor = 90.;
  float m_waveHeight = 90.;
  bool m_isRunning = true;
  int m_index = 0;
  int m_points = 0;
  int m_loses = 0;

  // Other game related content
  std::unique_ptr<Player> m_pplayer;

  std::vector<Obstacles> m_obstacles = {};
  void reviveObstacle(int index, float random, bool upsideDown);
  bool checkIntersection(const Player &player, const Obstacles &obstacle,
                         int index);
  void afterLosing();
  void clearObstacles();

  template <std::size_t T>
  glm::vec2 projection(const std::array<glm::vec2, T> &shape,
                       const glm::vec2 &axis);

public:
  ~Population()
  {
    delete m_obstacleX;
    delete m_obstacleY;
    delete m_playerY;
    delete m_playerVy;
    delete m_playerRot;
  }
};

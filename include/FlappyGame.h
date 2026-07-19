#pragma once
#include <pain.h>
#include <painless.h>

#include "Obstacles.h"
#include "Player.h"

class FlappyGame : public pain::WorldObject {
public:
  reg::Entity static create(pain::Scene &scene, pain::Application &app);

  void onCreate();
  void onUpdate(pain::DeltaTime deltaTime);
  void onRender(pain::RenderContext &renderer, 
                pain::DeltaTime currentTime);

  FlappyGame(reg::Entity entity, pain::Scene &scene, PlayerController *pc,
             pain::Material &om, std::vector<ObstaclesController *> obc,
             pain::Application &a);

protected:
  bool m_rendering = true;
  int m_panelID = -1;
  // parameters
  constexpr static int s_numberOfObstacles = 20;
  float m_obstaclesSpacing = 0.35f;
  float m_obstaclesInterval = 1.6f;
  float m_intervalTime = 0.6f;
  float m_maxInterval = 1.6f;
  float m_defaultObstacleSpeed = -0.32f;
  float m_colorInterval = 0.2f;  // color waves
  float m_heightInterval = 20.f; // height waves

  float m_waveColor = 90.f;
  float m_waveHeight = 90.f;
  bool m_isRunning = true;
  int m_index = 0;
  int m_points = 0;
  int m_loses = 0;

  PlayerController *m_playerController;

  std::vector<ObstaclesController *> m_obstacles = {};
  pain::Material &m_obstaclesMaterial;

  pain::Application &m_app;

  void changeObstaclesColors(pain::Color color);
  void reviveObstacle(int index, float random, bool upsideDown);
  bool checkIntersection(const ObstaclesController &obstacle);
  void afterLosing();
  void clearObstacles();

  std::tuple<PlayerController *, pain::Material &,
             std::vector<ObstaclesController
                             *>> static createHelper(pain::Scene &scene,
                                                     pain::Application &);

  template <std::size_t T>
  glm::vec2 projection(const std::array<glm::vec2, T> &shape,
                       const glm::vec2 &axis);
};

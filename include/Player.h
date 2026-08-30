#pragma once

#include <pain.h>
#include <painless.h>
#define DEFAULTXPOS -0.8f

struct ObstaclesController;

reg::Entity createPlayer(pain::Scene &scene, pain::RenderApi &renderAPI);

struct PlayerController : public pain::WorldObject {
public:
  PlayerController(reg::Entity entity, pain::Scene &scene);

  void onCreate();
  void onUpdate(pain::DeltaTime deltaTimeSec);
  void onRender(pain::RenderContext &renderer, pain::DeltaTime currentTime);
  void onDestroy();

  void resetPosition();

  bool checkIntersection(const ObstaclesController &obstacle);
  std::vector<ObstaclesController *> &getVisibleObstacles();

  // HACK: This exists because I can't figure out how push events
  // w/SDL_PushEvent
  bool m_automaticJump = false;

  std::string name = "undefined";

  static constexpr float MAX_HEIGHT = 1.f;
  static constexpr float MIN_HEIGHT = -1.f;

  std::vector<ObstaclesController *> m_obstacles = {};
  std::vector<ObstaclesController *> m_visibleObstacles;

private:
  // physics
  float m_pseudoVelocityX = 1.f;
  float m_maxVelY = 2.f;
  float m_gravity = -0.9f;
  float m_jumpForce = 0.0f;
  float m_jumpImpulse = 4.f;
  float m_dampingFactor = 1.f;
  // GUI
  bool m_displayUpdates = false;
  int m_panelID = -1;

  // particle emission
  float m_timeSinceLastEmission = 0.f;
  float m_emissionInterval = 0.02f;

  template <std::size_t T>
  glm::vec2 projection(const std::array<glm::vec2, T> &shape,
                       const glm::vec2 &axis);
};

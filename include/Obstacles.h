#pragma once
#include <pain.h>

#include "Core.h"
#include "Player.h"

class ObstaclesController : public pain::ScriptableEntity
{
public:
  void onUpdate(double deltaTime);
  void revive(bool upsideDown, bool movable);
  const void onImGuiUpdate();
  void setObstaclesSpeed();
  void revive(float obstacleSpeed, float height, bool upsideDown, int *points);
  void changeColor(glm::vec3 color);
  int *m_points = nullptr;

  ObstaclesController() = default;
  ~ObstaclesController() = default;
  MOVABLES(ObstaclesController)
  NONCOPYABLE(ObstaclesController)
private:
  bool m_isAlive = true;
  bool m_isUpsideDown = false;
  bool m_isMovable = true;
  bool m_canCountPoints = true;
  double m_deactivateTimout = 0.0f;
};

class Obstacles : public pain::GameObject
{
public:
  Obstacles(pain::Scene *scene);
  ~Obstacles() = default;
  // Obstacles &operator=(const Obstacles &o) { return *this; }
  // Obstacles(const Obstacles &o, pain::Scene *scene) : GameObject(scene) {}
  MOVABLES(Obstacles)
  NONCOPYABLE(Obstacles)
};

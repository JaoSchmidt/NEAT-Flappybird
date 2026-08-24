#pragma once
#include <pain.h>

#include "Core.h"

class ObstaclesController : public pain::WorldObject
{
public:
  using pain::WorldObject::WorldObject;
  reg::Entity static create(pain::Scene &scene, pain::Material &m);
  void onUpdate(pain::DeltaTime deltaTime);
  void revive(bool upsideDown, bool movable);
  void setObstaclesSpeed();
  void revive(float obstacleSpeed, float height, bool upsideDown, int *points);
  int *m_points = nullptr;

  ~ObstaclesController() = default;
  ObstaclesController(const ObstaclesController &) = delete;
  ObstaclesController &operator=(const ObstaclesController &) = delete;
  ObstaclesController(ObstaclesController &&) = default;
  ObstaclesController &operator=(ObstaclesController &&) = default;
  bool isUpsideDown() const { return m_isUpsideDown; }

private:
  bool m_isAlive = true;
  bool m_isUpsideDown = false;
  bool m_isMovable = true;
  bool m_canCountPoints = true;
  double m_deactivateTimout = 0.0f;
};

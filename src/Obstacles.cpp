#include "Obstacles.h"

Obstacles::Obstacles(pain::Scene *scene) : pain::GameObject(scene)
{
  addComponent<pain::TransformComponent>(glm::vec3(2.0f, -0.5f, 0.f));
  addComponent<pain::MovementComponent>();
  addComponent<pain::TrianguleComponent>(glm::vec2(0.8f, 2.00f),
                                         glm::vec4(0.2f, 0.3f, 0.9f, 1.0f));
}

void ObstaclesController::onUpdate(double deltaTime)
{
  const pain::TransformComponent &tc = getComponent<pain::TransformComponent>();
  if (tc.m_position.x < DEFAULTXPOS && m_isUpsideDown && m_canCountPoints) {
    (*m_points)++;
    m_canCountPoints = false;
  }
  // LOG_I("tc = ({},{},{})", TP_VEC3(tc.m_position));
}

void ObstaclesController::changeColor(glm::vec3 color)
{
  pain::TrianguleComponent &tgc = getComponent<pain::TrianguleComponent>();
  tgc.m_color = {color, 1.0f};
}

void ObstaclesController::revive(float obstacleSpeed, float height,
                                 bool upsideDown, int *points)
{
  pain::MovementComponent &mc = getComponent<pain::MovementComponent>();
  pain::TransformComponent &tc = getComponent<pain::TransformComponent>();
  pain::TrianguleComponent &tgc = getComponent<pain::TrianguleComponent>();
  m_points = points;
  tgc.m_color = {0.5f, 0.5f, 0.5f, 1.0f};
  m_isUpsideDown = upsideDown;
  if (m_isUpsideDown)
    tgc.m_height = {0.8f, -2.f};
  else
    tgc.m_height = {0.8f, 2.f};

  mc.m_velocityDir.x = obstacleSpeed;
  // WARN: This value "1.5f" to put all obstacles hidden on the right of the
  // screen might not work depending on the resolution. Consider alternatives
  tc.m_position = glm::vec3(2.f, height, 0.f);
  m_canCountPoints = true;
}

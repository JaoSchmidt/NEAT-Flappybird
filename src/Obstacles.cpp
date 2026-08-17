#include "Obstacles.h"
#include "Player.h"

reg::Entity ObstaclesController::create(pain::Scene &scene, pain::Material &m)
{

  reg::Entity e = scene.createEntity("Obstacle");
  scene.createComponents(
      e, //
      cmp::Pos2d(glm::vec3(2.0F, -0.5F, 0.F)), cmp::Mov2d(),
      cmp::Sprite::create({.layer = pain::RenderLayer::B,
                           .shape = pain::TriangleShape{0.8F, 2.00F}}),
      cmp::Material::create(m), //
      cmp::Script{});

  pain::Scene::emplaceScript<ObstaclesController>(e, scene);

  return e;
}

void ObstaclesController::onUpdate(pain::DeltaTime _)
{
  const cmp::Pos2d &tc = getComponent<cmp::Pos2d>();
  if (tc.m_position.x < DEFAULTXPOS && m_isUpsideDown && m_canCountPoints) {
    (*m_points)++;
    m_canCountPoints = false;
  }
}

void ObstaclesController::revive(float obstacleSpeed, float height,
                                 bool upsideDown, int *points)
{
  cmp::Mov2d &mc = getComponent<cmp::Mov2d>();
  cmp::Pos2d &tc = getComponent<cmp::Pos2d>();
  cmp::Sprite &sp = getComponent<cmp::Sprite>();
  pain::TriangleShape &ts = std::get<pain::TriangleShape>(sp.m_shape);
  m_points = points;
  // tgc.m_color = {0.5f, 0.5f, 0.5f, 1.0f};
  m_isUpsideDown = upsideDown;
  if (m_isUpsideDown)
    ts = {0.8F, -2.F};
  else
    ts = {0.8F, 2.F};

  mc.m_velocity.x = obstacleSpeed;
  // WARN: This value "1.5f" to put all obstacles hidden on the right of the
  // screen might not work depending on the resolution. Consider alternatives
  tc.m_position = glm::vec3(2.f, height, 0.f);
  m_canCountPoints = true;
}

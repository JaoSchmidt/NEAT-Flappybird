#include "Player.h"
#include "Obstacles.h"
#include "pain.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unistd.h>

reg::Entity createPlayer(pain::Scene &scene, pain::Material &m)
{

  reg::Entity entity = scene.createEntity("Player");
  scene.createComponents(entity, cmp::Pos2d{glm::vec2(DEFAULTXPOS, 0.0F)},
                         cmp::Mov2d{glm::vec2(0.F, 0.0F), 1.0F}, //
                         cmp::Material::create(m),               //
                         cmp::Script{},                          //
                         cmp::ParticleSpray::create({
                             .interval = pain::DeltaTime::oneSecond() / 8,
                             .randAngleFactor = 20.F,
                             .autoEmit = false,
                             .capacity = 100,
                         }),
                         cmp::Rot{315.F, glm::vec3(0.F, 1.F, 0.F)},          //
                         cmp::Sprite::create({.layer = pain::RenderLayer::B, //
                                              .shape = pain::RectShape{}})); //
  pain::Scene::emplaceScript<PlayerController>(entity, scene);
  return entity;
};

PlayerController::PlayerController(reg::Entity entity, pain::Scene &scene)
    : pain::WorldObject(entity, scene)
{
}

void PlayerController::onCreate()
{
  cmp::Mov2d &mc = getComponent<cmp::Mov2d>();
  mc.m_rotationSpeed = 0.0F;
  m_dampingFactor = 50.F;
  m_emissionInterval = 0.02F;
  cmp::ParticleSpray &psc = getComponent<cmp::ParticleSpray>();
  psc.lifeTime = pain::DeltaTime::oneMilliSecond() * 700;
  psc.randSizeFactor = 1.F;
  psc.sizeChangeSpeed = 0.15F;

  painless::customPanel::registerPanel("Controller", 1.F,
                                       painless::InterfaceMenu::SIDEBAR);
  painless::customPanel::addToPanel("Controller", [this]() {
    ImGui::Text("General Settings");
    ImGui::InputFloat("Gravity", &m_gravity, 0.01F, 1.0F, "%.3F");
    ImGui::InputFloat("Jump Impulse", &m_jumpImpulse, 0.1F, 1.0F, "%.3F");
    ImGui::InputFloat("Damping Effect", &m_dampingFactor, 0.01F, 1.0F, "%.5F");
    ImGui::InputFloat("Pseudo Velocity X", &m_pseudoVelocityX, 0.1F, 1.0F,
                      "%.3F");
    ImGui::SeparatorText("Log");
    ImGui::Checkbox("Log Updates", &m_displayUpdates);
    if (ImGui::Button("Reset Components"))
      resetPosition();
  });
}

void PlayerController::onRender(pain::RenderContext &renderAPI,
                                pain::DeltaTime currentTime)
{

  UNUSED(renderAPI)

  const cmp::Pos2d &tc = getComponent<cmp::Pos2d>();
  const cmp::Rot &rc = getComponent<cmp::Rot>();

  cmp::ParticleSpray &psc = getComponent<cmp::ParticleSpray>();
  const Uint8 *state = SDL_GetKeyboardState(NULL);
  if (state[SDL_SCANCODE_SPACE]) {

    // particles
    // Check if it's time to emit a new particle
    if (m_timeSinceLastEmission >= m_emissionInterval) {

      const float rando =
          static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5F;
      const float randoAngle = rando * glm::radians(psc.randAngleFactor);
      // rotation is already rotated 90 degrees btw
      const glm::mat2 rotation =
          glm::mat2(-sin(randoAngle), -cos(randoAngle), //
                    cos(randoAngle), -sin(randoAngle));

      pain::SprayParticle &p = psc.particles[psc.currentParticle];
      psc.next();
      p = {.offset = tc.m_position,
           .normal = glm::vec2(rotation * rc.m_rotation * 0.05F),
           .startTime = currentTime,
           .alive = true};
      m_timeSinceLastEmission = 0.0F; // Reset the timer
    }
  }
}

void PlayerController::onUpdate(pain::DeltaTime deltaTime)
{
  const float deltaTimeSec = deltaTime.getSecondsf();
  cmp::Pos2d &tc = getComponent<cmp::Pos2d>();
  cmp::Mov2d &mc = getComponent<cmp::Mov2d>();
  cmp::Rot &rc = getComponent<cmp::Rot>();

  if (m_jumpForce > 0.F)
    m_jumpForce = m_jumpForce - deltaTimeSec * m_dampingFactor;
  else
    m_jumpForce = 0.F;

  m_timeSinceLastEmission += deltaTimeSec;
  const Uint8 *state = SDL_GetKeyboardState(NULL);
  // LOG_I("scancode  {}", state[SDL_SCANCODE_SPACE]);
  if (state[SDL_SCANCODE_SPACE] || m_automaticJump) {
    m_jumpForce = m_jumpImpulse;
  }

  float acc;
  if (tc.m_position.y > MAX_HEIGHT) {
    tc.m_position.y = 1.F;
    acc = m_gravity * 10.F;
  } else if (tc.m_position.y < MIN_HEIGHT) {
    tc.m_position.y = -1.F;
    acc = m_jumpForce * 10.F;
  } else {
    acc = m_gravity + m_jumpForce;
  }

  // velocity y
  mc.m_velocity.y =
      std::clamp(mc.m_velocity.y + acc * deltaTimeSec, -m_maxVelY, m_maxVelY);
  // "velocity x"
  rc.m_rotationRadians =
      -std::numbers::pi / 2 + std::atan2(mc.m_velocity.y, m_pseudoVelocityX);

  if (state[SDL_SCANCODE_SPACE] && m_displayUpdates) {
    LOG_I("---------------------------------------------");
    LOG_I("m_jumpForce {}", m_jumpForce);
    LOG_I("acc {}", acc);
    LOG_I("mc.m_velocity.y {}", mc.m_velocity.y);
    LOG_I("Y/X {}", mc.m_velocity.y / m_pseudoVelocityX);
    LOG_I("atan {}", rc.m_rotationRadians);
  }
}

void PlayerController::resetPosition()
{
  cmp::Pos2d &tc = getComponent<cmp::Pos2d>();
  cmp::Mov2d &mc = getComponent<cmp::Mov2d>();
  cmp::Rot &rc = getComponent<cmp::Rot>();

  mc.m_rotationSpeed = 0.0F;
  m_pseudoVelocityX = 1.F;
  tc.m_position = {-0.8F, 0.F};
  mc.m_velocity = {0.F, 1.F};
  rc.m_rotation = {0.F, 1.F, 0.F};
  rc.m_rotationRadians = 315.F;
}

// void PlayerController::onDestroy() { delete m_pIG; }

template <std::size_t T>
glm::vec2 PlayerController::projection(const std::array<glm::vec2, T> &shape,
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

bool PlayerController::checkIntersection(const ObstaclesController &obstacle)
{
  auto &ptc = getComponent<cmp::Pos2d>();
  auto &prc = getComponent<cmp::Rot>();
  auto &psc = getComponent<cmp::Sprite>();
  auto &otc = obstacle.getComponent<cmp::Pos2d>();
  auto &osc = obstacle.getComponent<cmp::Sprite>();

  // get quad vertices
  constexpr glm::vec4 quadVertexPositions[4] = {
      glm::vec4(-0.5f, -0.5f, 0.f, 1.f),
      glm::vec4(0.5f, -0.5f, 0.f, 1.f),
      glm::vec4(0.5f, 0.5f, 0.f, 1.f),
      glm::vec4(-0.5f, 0.5f, 0.f, 1.f),
  };

  const pain::RectShape &qs = std::get<pain::RectShape>(psc.m_shape);
  const glm::mat4 transform = pain::Renderer2d::getTransform(
      ptc.m_position, qs.size, prc.m_rotationRadians);

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
  const pain::TriangleShape &ts = std::get<pain::TriangleShape>(osc.m_shape);
  const glm::mat4 transformTri =
      pain::Renderer2d::getTransform(otc.m_position, {ts.base, ts.height});
  const std::array<glm::vec2, 3> tVertices = {
      transformTri * triVertexPositions[0],
      transformTri * triVertexPositions[1],
      transformTri * triVertexPositions[2],
  };

  std::vector<glm::vec2> axes;
  for (size_t i = 0; i < 4; i++) {
    glm::vec2 edge = qVertices[(i + 1) % 4] - qVertices[i];
    glm::vec2 axis(-edge.y, edge.x);
    axis = glm::normalize(axis);
    axes.push_back(axis);
  }
  for (size_t i = 0; i < 3; i++) {
    glm::vec2 edge = tVertices[(i + 1) % 3] - tVertices[i];
    glm::vec2 axis(-edge.y, edge.x);
    axis = glm::normalize(axis);
    axes.push_back(axis);
  }
  for (const glm::vec2 &axis : axes) {
    auto boundA = projection(qVertices, axis);
    auto boundB = projection(tVertices, axis);
    if (boundA.y < boundB.x || boundB.y < boundA.x)
      return false;
  }
  return true;
}

int PlayerController::getClosestObstacle()
{
  const float &currentX =
      m_obstacles[m_closestObsIndex]->getComponent<cmp::Pos2d>().m_position.x;
  if (currentX != 2.f)
    return m_closestObsIndex;
  float closestX = 999999.f;
  int closestIndex = m_closestObsIndex;
  for (int i = 0; i < static_cast<int>(m_obstacles.size()); i++) {
    const float &x = m_obstacles[i]->getComponent<cmp::Pos2d>().m_position.x;
    if (x < closestX) {
      closestIndex = i;
      closestX = x;
    }
  }
  return closestIndex;
}

std::array<int, 2> PlayerController::getClosestObstacles()
{
  float closest1 = 999999.f;
  float closest2 = 999999.f;
  int idx1 = -1;
  int idx2 = -1;

  for (int i = 0; i < static_cast<int>(m_obstacles.size()); i++) {
    const float x = m_obstacles[i]->getComponent<cmp::Pos2d>().m_position.x;
    if (x < closest1) {
      closest2 = closest1;
      idx2 = idx1;
      closest1 = x;
      idx1 = i;
    } else if (x < closest2) {
      closest2 = x;
      idx2 = i;
    }
  }
  return {idx1, idx2};
}

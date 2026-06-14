#include "Player.h"
#include "pain.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unistd.h>

reg::Entity createPlayer(pain::Scene &scene, pain::Material &m,
                         painless::CustomEditor &ce) {

  reg::Entity entity = scene.createEntity();
  scene.createComponents(
      entity, pain::Transform2dComponent{glm::vec2(DEFAULTXPOS, 0.0f)},
      pain::Movement2dComponent{glm::vec2(0.f, 0.0f), 1.0f}, //
      pain::MaterialComponent::create(m),                    //
      pain::NativeScriptComponent{},                         //
      pain::ParticleSprayComponent::create({
          .interval = pain::DeltaTime::oneSecond() / 8,
          .randAngleFactor = 20.f,
          .autoEmit = false,
          .capacity = 100,
      }),
      pain::RotationComponent{315.f, glm::vec3(0.f, 1.f, 0.f)},          //
      pain::SpriteComponent::create({.layer = pain::RenderLayer::Closer, //
                                     .shape = pain::RectShape{}}));      //
  pain::Scene::emplaceScript<PlayerController>(entity, scene, ce);
  return entity;
};

PlayerController::PlayerController(reg::Entity entity, pain::Scene &scene,
                                   painless::CustomEditor &ce)
    : pain::WorldObject(entity, scene), m_customEditor(ce) {}

void PlayerController::onCreate() {
  pain::Movement2dComponent &mc = getComponent<pain::Movement2dComponent>();
  mc.m_rotationSpeed = 0.0f;
  m_pseudoVelocityX = 1.f;
  m_gravity = -0.9f;
  m_jumpForce = 0.0f;
  m_jumpImpulse = 4.0f;
  m_dampingFactor = 50.f;
  m_maxVelY = 2.f;
  m_displayUpdates = false;
  m_emissionInterval = 0.02f;
  pain::ParticleSprayComponent &psc =
      getComponent<pain::ParticleSprayComponent>();
  psc.lifeTime = pain::DeltaTime::oneMilliSecond() * 700;
  psc.randSizeFactor = 1.f;
  psc.sizeChangeSpeed = 0.15f;

  m_customEditor.registerPanel("Player Controller", 1.f,
                               painless::InterfaceMenu::SIDEBAR);
  m_customEditor.addToPanel("Player Controller", 1, [this]() {
    ImGui::Begin("Player Controller");
    ImGui::Text("General Settings");
    ImGui::InputFloat("Gravity", &m_gravity, 0.01f, 1.0f, "%.3f");
    ImGui::InputFloat("Jump Impulse", &m_jumpImpulse, 0.1f, 1.0f, "%.3f");
    ImGui::InputFloat("Damping Effect", &m_dampingFactor, 0.01f, 1.0f, "%.5f");
    ImGui::InputFloat("Pseudo Velocity X", &m_pseudoVelocityX, 0.1f, 1.0f,
                      "%.3f");
    ImGui::SeparatorText("Log");
    ImGui::Checkbox("Log Updates", &m_displayUpdates);
    if (ImGui::Button("Reset Components"))
      resetPosition();
    ImGui::End();
  });
}

void PlayerController::onRender(pain::RenderContext &renderers,
                                bool isMinimized, pain::DeltaTime currentTime) {

  UNUSED(renderers)
  UNUSED(isMinimized)
  constexpr glm::mat2 rotate90{glm::vec2(0, -1), glm::vec2(1, 0)};
  const pain::Transform2dComponent &tc =
      getComponent<pain::Transform2dComponent>();
  const pain::RotationComponent &rc = getComponent<pain::RotationComponent>();

  pain::ParticleSprayComponent &psc =
      getComponent<pain::ParticleSprayComponent>();
  const Uint8 *state = SDL_GetKeyboardState(NULL);
  if (state[SDL_SCANCODE_SPACE]) {

    // particles
    // Check if it's time to emit a new particle
    if (m_timeSinceLastEmission >= m_emissionInterval) {

      const float rando =
          static_cast<float>(rand()) / static_cast<float>(RAND_MAX) - 0.5f;
      const float randoAngle = rando * glm::radians(psc.randAngleFactor);
      // already rotated 90 degrees btw
      const glm::mat2 rotation =
          glm::mat2(-sin(randoAngle), -cos(randoAngle), //
                    cos(randoAngle), -sin(randoAngle));

      pain::SprayParticle &p = psc.particles[psc.currentParticle];
      p = {.offset = tc.m_position,
           .normal = glm::vec2(rotate90 * rc.m_rotation * 0.05f),
           .startTime = currentTime,
           .alive = true};
      m_timeSinceLastEmission = 0.0f; // Reset the timer
    }
  }
}

void PlayerController::onUpdate(pain::DeltaTime deltaTime) {
  double deltaTimeSec = deltaTime.getSeconds();
  pain::Transform2dComponent &tc = getComponent<pain::Transform2dComponent>();
  pain::Movement2dComponent &mc = getComponent<pain::Movement2dComponent>();
  pain::RotationComponent &rc = getComponent<pain::RotationComponent>();

  if (m_jumpForce > 0.f)
    m_jumpForce = m_jumpForce - deltaTimeSec * m_dampingFactor;
  else if (m_jumpForce <= 0.f)
    m_jumpForce = 0.f;

  m_timeSinceLastEmission += deltaTimeSec;
  const Uint8 *state = SDL_GetKeyboardState(NULL);
  // LOG_I("scancode  {}", state[SDL_SCANCODE_SPACE]);
  if (state[SDL_SCANCODE_SPACE] || m_automaticJump) {
    m_jumpForce = m_jumpImpulse;
  }

  float acc;
  if (tc.m_position.y > 1.f) {
    tc.m_position.y = 1.f;
    acc = m_gravity * 10.f;
  } else if (tc.m_position.y < -1.f) {
    tc.m_position.y = -1.f;
    acc = m_jumpForce * 10.f;
  } else {
    acc = m_gravity + m_jumpForce;
  }

  // velocity y
  mc.m_velocity.y = std::clamp(mc.m_velocity.y + acc * (float)deltaTimeSec,
                               -m_maxVelY, m_maxVelY);
  // "velocity x"
  rc.m_rotationAngle =
      -std::numbers::pi / 2 + std::atan2(mc.m_velocity.y, m_pseudoVelocityX);

  if (m_displayUpdates) {
    LOG_I("---------------------------------------------");
    LOG_I("m_jumpForce {}", m_jumpForce);
    LOG_I("acc {}", acc);
    LOG_I("mc.m_velocity.y {}", mc.m_velocity.y);
    LOG_I("Y/X {}", mc.m_velocity.y / m_pseudoVelocityX);
    LOG_I("atan {}", rc.m_rotationAngle);
  }
}

void PlayerController::resetPosition() {
  pain::Transform2dComponent &tc = getComponent<pain::Transform2dComponent>();
  pain::Movement2dComponent &mc = getComponent<pain::Movement2dComponent>();
  pain::RotationComponent &rc = getComponent<pain::RotationComponent>();

  mc.m_rotationSpeed = 0.0f;
  m_pseudoVelocityX = 1.f;
  tc.m_position = {-0.8f, 0.f};
  mc.m_velocity = {0.f, 1.f};
  rc.m_rotation = {0.f, 1.f, 0.f};
  rc.m_rotationAngle = 315.f;
}

// void PlayerController::onDestroy() { delete m_pIG; }

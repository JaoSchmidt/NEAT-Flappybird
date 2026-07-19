#include "Player.h"
#include "pain.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <unistd.h>

reg::Entity createPlayer(pain::Scene &scene, pain::Material &m) {

  reg::Entity entity = scene.createEntity();
  scene.createComponents(
      entity, pain::Transform2dComponent{glm::vec2(DEFAULTXPOS, 0.0F)},
      pain::Movement2dComponent{glm::vec2(0.F, 0.0F), 1.0F}, //
      pain::MaterialComponent::create(m),                    //
      pain::NativeScriptComponent{},                         //
      pain::ParticleSprayComponent::create({
          .interval = pain::DeltaTime::oneSecond() / 8,
          .randAngleFactor = 20.F,
          .autoEmit = false,
          .capacity = 100,
      }),
      pain::RotationComponent{315.F, glm::vec3(0.F, 1.F, 0.F)},     //
      pain::SpriteComponent::create({.layer = pain::RenderLayer::E, //
                                     .shape = pain::RectShape{}})); //
  pain::Scene::emplaceScript<PlayerController>(entity, scene);
  return entity;
};

PlayerController::PlayerController(reg::Entity entity, pain::Scene &scene)
    : pain::WorldObject(entity, scene) {}

void PlayerController::onCreate() {
  pain::Movement2dComponent &mc = getComponent<pain::Movement2dComponent>();
  mc.m_rotationSpeed = 0.0F;
  m_dampingFactor = 50.F;
  m_emissionInterval = 0.02F;
  pain::ParticleSprayComponent &psc =
      getComponent<pain::ParticleSprayComponent>();
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

void PlayerController::onRender(pain::RenderContext &renderers,
                                 pain::DeltaTime currentTime) {

  UNUSED(renderers)
  
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

void PlayerController::onUpdate(pain::DeltaTime deltaTime) {
  const float deltaTimeSec = deltaTime.getSecondsf();
  pain::Transform2dComponent &tc = getComponent<pain::Transform2dComponent>();
  pain::Movement2dComponent &mc = getComponent<pain::Movement2dComponent>();
  pain::RotationComponent &rc = getComponent<pain::RotationComponent>();

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

void PlayerController::resetPosition() {
  pain::Transform2dComponent &tc = getComponent<pain::Transform2dComponent>();
  pain::Movement2dComponent &mc = getComponent<pain::Movement2dComponent>();
  pain::RotationComponent &rc = getComponent<pain::RotationComponent>();

  mc.m_rotationSpeed = 0.0F;
  m_pseudoVelocityX = 1.F;
  tc.m_position = {-0.8F, 0.F};
  mc.m_velocity = {0.F, 1.F};
  rc.m_rotation = {0.F, 1.F, 0.F};
  rc.m_rotationRadians = 315.F;
}

// void PlayerController::onDestroy() { delete m_pIG; }

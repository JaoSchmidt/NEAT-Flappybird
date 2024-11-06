#pragma once

#include <pain.h>
#define DEFAULTXPOS -0.8f

class PlayerController : public pain::ScriptableEntity,
                         public pain::ImGuiInstance
{
public:
  void onCreate();
  void onUpdate(double deltaTimeSec);
  void onRender(double currentTime);
  const void onImGuiUpdate() override;

  const void resetPosition();

  // HACK: This exists because I can figure out how push events w/SDL_PushEvent
  bool m_automaticJump = false;

private:
  float m_pseudoVelocityX = 1.f;
  float m_maxVelY = 2.f;
  float m_gravity = -0.9f;
  float m_jumpForce = 1.0f;
  float m_jumpImpulse = 5.f;
  float m_dampingFactor = 1.f;
  bool m_displayUpdates = false;
  bool m_isRendering = false;
  // particle emission
  float m_timeSinceLastEmission = 0.f;
  float m_emissionInterval = 0.f;
  // NEAT automation
};

class Player : public pain::GameObject
{
public:
  Player(pain::Scene *scene, std::shared_ptr<pain::Texture> &pTexture);
};

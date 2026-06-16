#pragma once

#include <pain.h>
#include <painless.h>
#define DEFAULTXPOS -0.8f

reg::Entity createPlayer(pain::Scene &scene, pain::Material &m,
                         painless::CustomEditor &customEditor);

struct PlayerController : public pain::WorldObject {
public:
  PlayerController(reg::Entity entity, pain::Scene &scene,
                   painless::CustomEditor &ce);
  void onCreate();
  void onUpdate(pain::DeltaTime deltaTimeSec);
  void onRender(pain::RenderContext &renderer, bool isMinimized,
                pain::DeltaTime currentTime);

  void resetPosition();

  // HACK: This exists because I can figure out how push events w/SDL_PushEvent
  bool m_automaticJump = false;

  std::string name = "undefined";

private:
  float m_pseudoVelocityX = 1.f;
  float m_maxVelY = 2.f;
  float m_gravity = -0.9f;
  float m_jumpForce = 0.0f;
  float m_jumpImpulse = 4.f;
  float m_dampingFactor = 1.f;
  bool m_displayUpdates = false;
  // bool m_isRendering = false;
  // particle emission
  float m_timeSinceLastEmission = 0.f;
  float m_emissionInterval = 0.02f;
  // NEAT automation

  painless::CustomEditor &m_customEditor;
};

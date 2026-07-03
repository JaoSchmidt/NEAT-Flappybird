/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "Others/MousePointer.h"
#include "ECS/Scriptable.h"
#include "SDL_events.h"
#include "imgui.h"
#include <pain.h>
#include <painless.h>

reg::Entity MousePointer::create(pain::Scene &scene, pain::Renderers &renderers,
                                 reg::Entity cameraEntity) {
  reg::Entity entity = scene.createEntity();

  pain::Texture &texture =
      pain::TextureManager::createTexture("resources/textures/cross.png");
  scene.createComponents(
      entity, pain::Transform2dComponent{},
      pain::MaterialComponent::create(
          renderers.m_materialManager.createMaterial(
              "MousePointer",
              {.params = pain::ParamSimplest{},
               .shader = renderers.m_materialManager.getDefaultShader(
                   pain::DefaultShader::Texture),
               .texture = texture})                                //
          ),                                                       //
      pain::SpriteComponent::create({.shape = pain::RectShape{}}), //
      pain::NativeScriptComponent{}                                //
  );

  pain::Scene::emplaceScript<MousePointer>(entity, scene, cameraEntity);
  return entity;
}

MousePointer::MousePointer(reg::Entity entity, pain::Scene &scene,
                           reg::Entity cameraEntity)
    : pain::WorldObject(entity, scene), m_cameraEntity(cameraEntity) {};

void MousePointer::onCreate() {
  const cmp::OrthoCamera &camCC =
      getComponent<cmp::OrthoCamera>(m_cameraEntity);
  PLOG_I("Resolution x {}", camCC.getResolution().x);
  PLOG_I("Resolution y {}", camCC.getResolution().y);
  PLOG_I("Zoom Level {}", camCC.m_zoomLevel);
  painless::customPanel::registerPanel("Mouse", 1.F,
                                       painless::InterfaceMenu::SIDEBAR);
  m_worldPosPanel = painless::customPanel::addToPanel(
      "Mouse", [=]() { ImGui::Text("World position (%.3f, %.3f)", 0.f, 0.f); });
}

glm::vec2 MousePointer::screenToWorld(int x, int y) {
  if (hasAnyComponents<pain::RotationComponent>(m_cameraEntity)) {
    const auto &[camCC, camTC, camRC] =
        getComponents<cmp::OrthoCamera, pain::Transform2dComponent,
                      pain::RotationComponent>(m_cameraEntity);
    return camCC.screenToWorld(x, y, camTC, camRC);
  }
  const auto &[camCC, camTC] =
      getComponents<cmp::OrthoCamera, pain::Transform2dComponent>(
          m_cameraEntity);
  return camCC.screenToWorld(x, y, camTC);
}

void MousePointer::onUpdate(pain::DeltaTime _) {
  int x = -1, y = -1;
  SDL_GetMouseState(&x, &y);
  pain::Transform2dComponent &tc = getComponent<pain::Transform2dComponent>();
  tc.m_position = screenToWorld(x, y);
  painless::customPanel::updateSubPanel("Mouse", m_worldPosPanel, [=]() {
    ImGui::Text("World position (%.3f, %.3f)", TP_VEC2(tc.m_position));
  });
}

void MousePointer::onEvent(const SDL_Event &event) {
  if (event.type == SDL_MOUSEMOTION) {
    pain::Transform2dComponent &tc = getComponent<pain::Transform2dComponent>();
    tc.m_position = screenToWorld(event.motion.x, event.motion.y);
    painless::customPanel::updateSubPanel("Mouse", m_worldPosPanel, [=]() {
      ImGui::Text("World position (%.3f, %.3f)", TP_VEC2(tc.m_position));
    });
  } else if (event.type == SDL_MOUSEWHEEL) {
    pain::Transform2dComponent &tc = getComponent<pain::Transform2dComponent>();
    tc.m_position = screenToWorld(event.wheel.mouseX, event.wheel.mouseY);
    painless::customPanel::updateSubPanel("Mouse", m_worldPosPanel, [=]() {
      ImGui::Text("World position (%.3f, %.3f)", TP_VEC2(tc.m_position));
    });
  }
}

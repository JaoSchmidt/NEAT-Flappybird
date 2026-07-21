/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "Others/MousePointer.h"
#include "ECS/Scriptable.h"
#include "SDL_events.h"
#include "imgui.h"
#include "imgui_internal.h"
#include <pain.h>
#include <painless.h>

reg::Entity MousePointer::create(pain::Scene &scene, pain::RenderApi &renderAPI,
                                 reg::Entity cameraEntity) {
  reg::Entity entity = scene.createEntity("MousePointer");

  pain::Texture &texture =
      pain::TextureManager::createTexture("resources/textures/cross.png");
  scene.createComponents(
      entity, pain::Transform2dComponent{},
      pain::MaterialComponent::create(
          renderAPI.m_materialManager.createMaterial(
              "MousePointer",
               {.shader = renderAPI.m_shaderManager.getDefaultShader(
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
    : pain::WorldObject(entity, scene), //
      m_cameraEntity(cameraEntity) {};

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

void MousePointer::onMouseButtonUp(const SDL_Event &event) {
  if (event.button.button == SDL_BUTTON_LEFT) {
    auto [tc, cc] = getComponents<pain::Transform2dComponent, cmp::OrthoCamera>(
        m_cameraEntity);
    pain::Transform2dComponent &mptc =
        getComponent<pain::Transform2dComponent>();
    PLOG_I("--------------------------------------------------------");
    PLOG_I("Camera position = ({},{})", TP_VEC2(tc.m_position));
    PLOG_I("Mouse Pointer Position = ({},{})", TP_VEC2(mptc.m_position));
    PLOG_I("resolution = ({},{})", TP_VEC2(cc.getResolution()));
  }
}

glm::vec2 MousePointer::screenToWorld(int x, int y) {
  int adjX = x, adjY = y;

  if (ImGui::GetCurrentContext() != nullptr) {
    ImGuiWindow *viewportWindow = ImGui::FindWindowByName("Viewport");
    if (viewportWindow != nullptr) {
      ImVec2 mainPos = ImGui::GetMainViewport()->Pos;
      ImVec2 contentMin = viewportWindow->ContentRegionRect.Min;
      float offsetX = contentMin.x - mainPos.x;
      float offsetY = contentMin.y - mainPos.y;
      adjX = x - static_cast<int>(offsetX);
      adjY = y - static_cast<int>(offsetY);
    }
  }

  if (hasAnyComponents<pain::RotationComponent>(m_cameraEntity)) {
    const auto &[camCC, camTC, camRC] =
        getComponents<cmp::OrthoCamera, pain::Transform2dComponent,
                      pain::RotationComponent>(m_cameraEntity);
    return camCC.screenToWorld(adjX, adjY, camTC, camRC);
  }
  const auto &[camCC, camTC] =
      getComponents<cmp::OrthoCamera, pain::Transform2dComponent>(
          m_cameraEntity);
  return camCC.screenToWorld(adjX, adjY, camTC);
}

void MousePointer::onUpdate(pain::DeltaTime _) {
  int x = 0, y = 0;
  SDL_GetMouseState(&x, &y);
  pain::Transform2dComponent &tc = getComponent<pain::Transform2dComponent>();
  tc.m_position = screenToWorld(x, y);
  painless::customPanel::updateSubPanel("Mouse", m_worldPosPanel, [=]() {
    ImGui::Text("World position (%.3f, %.3f)", TP_VEC2(tc.m_position));
  });
}

void MousePointer::onEvent(const SDL_Event &event) {
  switch (event.type) {
  case SDL_MOUSEBUTTONUP:
    onMouseButtonUp(event);
    break;
  case SDL_MOUSEMOTION: {
    pain::Transform2dComponent &tc = getComponent<pain::Transform2dComponent>();
    tc.m_position = screenToWorld(event.motion.x, event.motion.y);
    painless::customPanel::updateSubPanel("Mouse", m_worldPosPanel, [=]() {
      ImGui::Text("World position (%.3f, %.3f)", TP_VEC2(tc.m_position));
    });
    break;
  }
  case SDL_MOUSEWHEEL: {
    pain::Transform2dComponent &tc = getComponent<pain::Transform2dComponent>();
    tc.m_position = screenToWorld(event.wheel.mouseX, event.wheel.mouseY);
    painless::customPanel::updateSubPanel("Mouse", m_worldPosPanel, [=]() {
      ImGui::Text("World position (%.3f, %.3f)", TP_VEC2(tc.m_position));
    });
    break;
  }
  default:
    return;
  }
}

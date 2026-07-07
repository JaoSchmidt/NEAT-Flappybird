/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#pragma once

#include "Core.h"
#include "ECS/Scriptable.h"

class MousePointer : public pain::WorldObject {
public:
  reg::Entity static create(pain::Scene &scene, pain::Renderers &renderers,
                            reg::Entity cameraEntity);
  MousePointer(reg::Entity entity, pain::Scene &scene,
               reg::Entity cameraEntity);

  void onEvent(const SDL_Event &e);
  void onUpdate(pain::DeltaTime deltaTime);
  void onCreate();

private:
  void onMouseButtonUp(const SDL_Event &event);
  int m_worldPosPanel = -1;
  reg::Entity m_cameraEntity = reg::Entity{-2};
  glm::vec2 screenToWorld(int mouseX, int mouseY);
};

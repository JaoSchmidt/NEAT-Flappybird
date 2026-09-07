#include "TwoTileScroll.h"
#include "Assets/ManagerTexture.h"

#include <algorithm>

#define BACKGROUND_TEXTURE_PATH "resources/textures/background_texture.png"

void TwoTileScroll::init(pain::RenderApi &renderAPI, float screenWidth,
                         float scrollSpeed, std::size_t numTiles)
{
  m_screenWidth = screenWidth;
  m_numTiles = std::max<std::size_t>(numTiles, 1);
  // Cover several full screens wide so the wrapping never exposes an empty gap.
  m_tileWidth = 1.1f;
  m_speed = scrollSpeed;

  pain::TextureManager::createTexture(BACKGROUND_TEXTURE_PATH);
  pain::Shader &backgroundShader =
      renderAPI.m_shaderManager.getDefaultShader(pain::DefaultShader::Texture);
  pain::Material &backgroundMaterial =
      renderAPI.m_materialManager.createMaterial(
          "Background Material", //
          pain::MaterialCreationInfo{
              .color = pain::Colors::FullWhite,
              .params = std::monostate{},
              .shader = backgroundShader,
          });
  // clamp=false keeps GL_REPEAT so the seamless texture wraps correctly.
  backgroundMaterial.setTexture(BACKGROUND_TEXTURE_PATH);
  m_material = &backgroundMaterial;
}

void TwoTileScroll::update(pain::DeltaTime deltaTime)
{
  m_scrollX += m_speed * deltaTime.getSecondsf();
  // Once one tile width has scrolled past the left edge the layout is
  // pixel identical, so we wrap back to a full-width offset.
  while (m_scrollX <= -m_tileWidth)
    m_scrollX += m_tileWidth;
}

void TwoTileScroll::render(pain::RenderContext &ctx) const
{
  if (m_material == nullptr)
    return;

  const float w = m_tileWidth;
  // Left edge of the left tile; the layout always covers the visible screen.
  const float x0 = -m_screenWidth * 0.5 + m_scrollX;
  const glm::vec2 size(w, 2.1f);

  for (std::size_t i = 0; i < m_numTiles; ++i)
    ctx.submitRect(glm::vec2(x0 + w * (0.5f + static_cast<float>(i)), 0.f),
                   size, m_layer, *m_material);
}

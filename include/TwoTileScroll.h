#pragma once

#include <cstddef>
#include <pain.h>

/**
 * @brief N-tile horizontally seamless scrolling background.
 *
 * Draws the same texture N times side by side. All tiles move left in world
 * space every update, and once one full tile width has scrolled past the left
 * edge the scroll offset wraps. Because the tile width wraps to an identical
 * screen layout, this produces an infinite, seam-free background without
 * needing any shader/UV support.
 */
class TwoTileScroll
{
public:
  /**
   * @brief Loads the background texture and creates its material.
   *
   * @param renderAPI        Engine render API used to create the material.
   * @param screenHalfWidth  Half of the visible world width (camera aspect).
   * @param scrollSpeed      World units per second. Negative scrolls left.
   * @param numTiles         Number of texture tiles to draw side by side.
   */
  void init(pain::RenderApi &renderAPI, float screenHalfWidth,
            float scrollSpeed = -0.35f, std::size_t numTiles = 2);

  /** @brief Advances the scroll offset by the given frame delta. */
  void update(pain::DeltaTime deltaTime);

  /** @brief Submits all tiles to the render context. */
  void render(pain::RenderContext &ctx) const;

private:
  float m_screenWidth = 1.f;  ///< Half of the visible world width.
  float m_height = 2.f;       ///< World height of each tile.
  float m_tileWidth = 3.f;    ///< World width of a single tile.
  std::size_t m_numTiles = 2; ///< Number of texture tiles side by side.
  float m_scrollX = 0.f;      ///< Scrolling offset, wraps in [-tileWidth, 0].
  float m_speed = -0.35f;     ///< Scroll speed in world units/second.
  pain::Material *m_material = nullptr;
  pain::RenderLayer m_layer = pain::RenderLayer::A;
};

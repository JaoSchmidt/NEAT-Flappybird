#include "NEAT/NN.h"

class GraphRender : public pain::WorldObject
{
  struct Layer {
    std::map<int, glm::vec2> m_coord;
    std::vector<int> m_nodes;
    Layer(int layer, std::vector<int> nodes);
    bool contains(int node) const { return m_coord.contains(node); }
    const glm::vec2 &getCoord(int node) const { return m_coord.at(node); };
  };

public:
  reg::Entity static create(pain::Scene &scene, pain::RenderApi &renderAPI,
                            reg::Entity camEntity);

  void onEvent(const SDL_Event &event);
  // void onRender(pain::RenderContext &renderer,
  //               pain::DeltaTime currentTime);
  GraphRender(reg::Entity entity, pain::Scene &scene, reg::Entity camEntity,
              pain::Font *font, const glm::vec2 &center = {0, 0});
  /**
   * Generates the visual graph representing the genome neural network of the
   * individual
   */
  void generateGraph(pain::Scene &scene,
                     const std::vector<ConnectionGene> &links,
                     const std::vector<std::string> &inputNames,
                     pain::Application &app);

  void updateWeights(const std::unordered_map<int, NodeInput> &weights);

private:
  std::map<int, reg::Entity> m_mapNodeEntity;
  std::vector<reg::Entity> m_circles;
  std::vector<reg::Entity> m_lines;
  std::vector<reg::Entity> m_misc;
  glm::vec2 m_centerCache = glm::vec2(0);

  // graph
  int m_numNodes = 0;
  int m_numEdges = 0;
  double m_maxWeight;
  double m_minWeight;

  // rendering
  bool m_dragging = false;

  glm::vec2 m_dragOffset{};
  glm::vec2 m_lastMouseWorld{};

  pain::Font *m_font = nullptr;

  reg::Entity m_camEntity;

  std::array<pain::Color, 16> m_palette = pain::Colors::makeGradient<16>(
      pain::Colors::PastelGreen, pain::Colors::PastelRed);
  static constexpr float RESIZE_MARGIN = 0.05f;
  glm::vec2 screenToWorld(int x, int y);
};

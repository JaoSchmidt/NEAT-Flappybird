#include "NN.h"

class GraphRender : public pain::WorldObject {
  struct Layer {
    std::map<int, glm::vec2> m_coord;
    std::vector<int> m_nodes;
    Layer(int layer, std::vector<int> nodes);
    bool contains(int node) const { return m_coord.contains(node); }
    const glm::vec2 &getCoord(int node) const { return m_coord.at(node); };
  };

public:
  reg::Entity static create(pain::Scene &scene, pain::Renderers &renderers);

  void onUpdate(pain::DeltaTime deltaTime);
  void onEvent(const SDL_Event &event);
  GraphRender(reg::Entity entity, pain::Scene &scene,
              pain::Material &nodeMaterial, pain::Material &lineMaterial);

  /**
   * Generates the visual graph representing the genome neural network of the
   * individual
   */
  void generateGraph(pain::Scene &scene,
                     const std::vector<ConnectionGene> &links);

private:
  std::vector<Layer> m_layers;
  std::map<const glm::vec2 *, const glm::vec2 *> m_lineCoordMap;
  std::vector<reg::Entity> m_circles;
  std::vector<reg::Entity> m_lines;

  // graph
  pain::Material &m_nodeMaterial;
  pain::Material &m_lineMaterial;
  int m_numNodes = 0;
  int m_numEdges = 0;

  // rendering
  bool m_dragging = false;

  glm::vec2 m_dragOffset{};
  glm::vec2 m_lastMouseWorld{};

  static constexpr float RESIZE_MARGIN = 0.05f;
};

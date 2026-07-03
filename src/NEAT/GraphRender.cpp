#include "NEAT/GraphRender.h"
#include "CoreRender/Buffers/Material.h"
#include "ECS/Registry/Entity.h"
#include <pain.h>

constexpr float SPACE_BETWEEN_LAYERS = 0.3F;
constexpr float SPACE_BETWEEN_NODES = 0.125F;
constexpr float LINK_THICKNESS = 0.01F;
constexpr float NODE_DIAMETER = 0.1F;

// score, deaths, generation, species

GraphRender::Layer::Layer(int layer, std::vector<int> nodes)
    : m_nodes(std::move(nodes)) {
  constexpr float n = SPACE_BETWEEN_NODES;
  constexpr float l = SPACE_BETWEEN_LAYERS;
  int size = static_cast<int>(m_nodes.size());
  float layerf = static_cast<float>(layer);

  for (int i = 0; i < static_cast<int>(m_nodes.size()); i++) {
    const int node = m_nodes[i];
    m_coord[node] = {layerf * l, -n / 2 * (size - 1) + i * n};
  }
}

reg::Entity GraphRender::create(pain::Scene &scene,
                                pain::Renderers &renderers) {
  const int w = 1024;
  const int h = 768;
  const float zoom = 1.f;
  const glm::vec2 center{-1.f, -1.f};

  reg::Entity entity = scene.createEntity();
  pain::Shader &nodeShader = renderers.m_materialManager.getDefaultShader(
      pain::DefaultShader::Circles);

  pain::Shader &lineShader = renderers.m_materialManager.loadShaderFromFile(
      "LineGraphShader", "resources/shaders/graphLine.glsl");
  pain::Material &nodeMaterial = renderers.m_materialManager.createMaterial(
      "GraphNodes", //
      pain::MaterialCreationInfo{
          .color = pain::Colors::SkyBlue,
          .params = pain::ParamSimplest{},
          .shader = nodeShader,
      } //
  );
  pain::Material &backGroundMaterial =
      renderers.m_materialManager.createMaterial(
          "GraphBackground", //
          pain::MaterialCreationInfo{
              .color = pain::Colors::Brown,
              .params = pain::ParamSimplest{},
              .shader = renderers.m_materialManager.getDefaultShader(
                  pain::DefaultShader::Texture),
          } //
      );
  pain::Material &lineMaterial = renderers.m_materialManager.createMaterial(
      "GraphLines", //
      pain::MaterialCreationInfo{
          .color = pain::Colors::Brown,
          .params = pain::ParamSimplest{},
          .shader = lineShader,
      } //
  );

  scene.createComponents(
      entity, pain::Transform2dComponent{}, pain::NativeScriptComponent{},
      pain::MaterialComponent::create(backGroundMaterial),
      pain::SpriteComponent::create({
          .layer = pain::RenderLayer::C,
          .shape = pain::RectShape({2.f, 1.f}),
      }),
      Component::OrthoCamera::create(true, w, h, zoom, entity) //
  );                                                           //
  //
  pain::Scene::emplaceScript<GraphRender>(entity, scene, nodeMaterial,
                                          lineMaterial);
  return entity;
}

void GraphRender::onEvent(const SDL_Event &event) {
  const pain::SpriteComponent &sprite = getComponent<pain::SpriteComponent>();
  auto [transform, camera] =
      getComponents<pain::Transform2dComponent, cmp::OrthoCamera>();

  glm::vec2 mouse =
      camera.screenToWorld(event.button.x, event.button.y, transform);
  PLOG_I("Mouse to world = ({},{})", TP_VEC2(mouse));
  PLOG_E("Mouse SDL = ({},{})", TP_VEC2(event.button));
  switch (event.type) {
  case SDL_MOUSEBUTTONDOWN: {
    if (event.button.button != SDL_BUTTON_LEFT)
      break;
    const pain::RectShape &rect = std::get<pain::RectShape>(sprite.m_shape);
    PLOG_I("Button down");

    glm::vec2 half = rect.size * 0.5f;

    glm::vec2 min = transform.m_position - half;
    glm::vec2 max = transform.m_position + half;
    // if outside, break
    if (mouse.x < min.x || mouse.x > max.x || mouse.y < min.y ||
        mouse.y > max.y) {
      PLOG_I("Mouse = ({},{}) and bottom-left = ({},{}), top-right = ({},{})",
             TP_VEC2(mouse), min.x, min.y, max.x, max.y);
      break;
    }

    m_dragging = true;
    PLOG_I("m_dragOffset = ({},{})", TP_VEC2(m_dragOffset));
    break;
  }

  case SDL_MOUSEBUTTONUP: {
    if (event.button.button == SDL_BUTTON_LEFT)
      m_dragging = false;
    break;
  }

  case SDL_MOUSEMOTION: {
    glm::vec2 mouse =
        camera.screenToWorld(event.motion.x, event.motion.y, transform);
    if (m_dragging) {
      transform.m_position = mouse;
    }
    break;
  }

  default:
    break;
  }
}
void GraphRender::onUpdate(pain::DeltaTime _) {

  pain::SpriteComponent &sprite = getComponent<pain::SpriteComponent>();
  pain::RectShape &rect = std::get<pain::RectShape>(sprite.m_shape);
  const Uint8 *state = SDL_GetKeyboardState(NULL);
  if (state[SDL_SCANCODE_J])
    rect.size.y -= 0.05f;
  if (state[SDL_SCANCODE_K])
    rect.size.y += 0.05f;
  if (state[SDL_SCANCODE_H])
    rect.size.x -= 0.05f;
  if (state[SDL_SCANCODE_L])
    rect.size.x += 0.05f;
}

GraphRender::GraphRender(reg::Entity entity, pain::Scene &scene,
                         pain::Material &nodeMaterial,
                         pain::Material &lineMaterial)
    : pain::WorldObject(entity, scene), m_nodeMaterial(nodeMaterial),
      m_lineMaterial(lineMaterial) {};

void GraphRender::generateGraph(pain::Scene &scene,
                                const std::vector<ConnectionGene> &links) {
  std::vector<int> currentInput = {-1, -2, -3, -4, -5};
  // Couple of things to know to help create a beautiful graph:
  // 1. the genome will already be sorted from the first to last layer
  // 2. nodes will always have at least 1 link behind them, unless they are
  // the input
  // 2. nodes will always have at least 1 link after them, unless they are
  // the output
  // 3. inputs nodes don't have links behind
  // 4. output nodes don't have links forward

  // =====================================================
  // clear previous genome
  for (reg::Entity entity : m_circles) {
    scene.removeEntity(entity);
  }
  for (reg::Entity entity : m_lines) {
    scene.removeEntity(entity);
  }
  m_numNodes = 0;
  m_numEdges = 0;
  m_layers.clear();
  m_lineCoordMap.clear();

  // =====================================================
  // Step 1 BFS:
  std::map<int, std::vector<int>> outgoing;
  std::map<int, int> inDegree;
  for (const auto &link : links) {
    m_numEdges++;
    outgoing[link.m_InNodeId].push_back(link.m_OutNodeId);
    inDegree[link.m_OutNodeId]++;
  }

  // Step 1:
  while (!currentInput.empty()) {
    int size = static_cast<int>(m_layers.size());
    Layer &layer = m_layers.emplace_back(size, std::move(currentInput));
    std::vector<int> next;
    for (int node : layer.m_nodes) {
      for (int destinationNode : outgoing[node]) {
        if (--inDegree[destinationNode] == 0) {
          next.push_back(destinationNode);
          m_numNodes++;
        }
      }
    }
    currentInput = std::move(next);
  }

  // map links from layer to layer. Graph is acyclical, i.e., no need to
  // test previous layers
  for (unsigned i = 0; i < m_layers.size(); i++) {
    const Layer &layer = m_layers[i];
    for (int node : layer.m_nodes) {
      for (unsigned j = i + 1; j < m_layers.size(); j++) {
        const Layer &olayer = m_layers[j];
        for (int destinationNode : outgoing[node]) {
          if (olayer.contains(destinationNode)) {
            m_lineCoordMap[&layer.getCoord(node)] =
                &olayer.getCoord(destinationNode);
          }
        }
      }
    }
  }

  // =====================================================
  // Draw everything:
  m_circles.reserve(m_numNodes);
  m_lines.reserve(m_numEdges);
  for (const Layer &layer : m_layers) {
    // circles (nodes)
    for (int node : layer.m_nodes) {
      reg::Entity entity = scene.createEntity();
      scene.createComponents(
          entity,                                           //
          pain::Transform2dComponent{layer.getCoord(node)}, //
          pain::SpriteComponent::create(
              {.layer = pain::RenderLayer::E,
               .shape = pain::QuadShape{NODE_DIAMETER}}),   //
          pain::MaterialComponent::create(m_nodeMaterial)); //
    }
  }
  // lines (edges)
  for (const auto [orig, dest] : m_lineCoordMap) {
    reg::Entity entity = scene.createEntity();
    scene.createComponents(
        entity,                            //
        pain::Transform2dComponent{*orig}, //
        pain::SpriteComponent::create(
            {.layer = pain::RenderLayer::D,
             .shape = pain::LineShape{*dest, LINK_THICKNESS}}), //
        pain::MaterialComponent::create(m_lineMaterial)         //
    );                                                          //
  }
}

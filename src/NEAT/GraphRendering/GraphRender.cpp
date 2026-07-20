#include "NEAT/GraphRendering/GraphRender.h"
#include "Core.h"
#include "NEAT/GraphRendering/GraphicNode.h"
#include "imgui.h"
#include "imgui_internal.h"
#include <pain.h>

constexpr float SPACE_BETWEEN_LAYERS = 0.3F;
constexpr float SPACE_BETWEEN_NODES = 0.125F;
constexpr float MAX_LINK_THICKNESS = 0.04F;
constexpr float MIN_LINK_THICKNESS = 0.005F;
constexpr float NODE_DIAMETER = 0.1F;

constexpr std::string_view materialNodes = "GraphNodes";
constexpr std::string_view materialLine = "GraphLines";
constexpr std::string_view materialBackground = "GraphBackground";
constexpr std::string_view materialTemp = "GraphTemp";

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

reg::Entity GraphRender::create(pain::Scene &scene, pain::RenderApi &renderAPI,
                                reg::Entity camEntity) {
  const float zoom = 1.f;
  const glm::vec2 center{-1.f, -1.f};

  reg::Entity entity = scene.createEntity();
  // pain::Shader &nodeShader = renderAPI.m_shaderManager.getDefaultShader(
  //     pain::DefaultShader::Circles);
  pain::Shader &backGroundShader = renderAPI.m_shaderManager.loadShaderFromFile(
      "GraphFrame", "resources/shaders/graphFrame.glsl");
  // pain::Shader &nodeShader =
  //     renderAPI.m_shaderManager.getDefaultShader(pain::DefaultShader::Texture);
  pain::Shader &nodeShader = renderAPI.m_shaderManager.loadShaderFromFile(
      "GraphNodeShader", "resources/shaders/graphNodes.glsl");
  pain::Shader &lineShader =
      renderAPI.m_shaderManager.getDefaultShader(pain::DefaultShader::Texture);
  pain::Font &font = renderAPI.m_fontManager.getDefault();

  // pain::Shader &lineShader = renderAPI.m_shaderManager.loadShaderFromFile(
  //     "LineGraphShader", "resources/shaders/graphLine.glsl");
  renderAPI.m_materialManager.createMaterial(
      materialNodes, //
      pain::MaterialCreationInfo{
          .color = pain::Colors::FullWhite,
          .shader = nodeShader,
      } //
  );
  pain::Material &backGround = renderAPI.m_materialManager.createMaterial(
      materialBackground, //
      pain::MaterialCreationInfo{
          .color = pain::Colors::StrongPink,
          .shader = backGroundShader,
      } //
  );
  pain::Material &temp = renderAPI.m_materialManager.createMaterial(
      materialTemp, //
      pain::MaterialCreationInfo{
          .color = pain::Colors::TransparentWhite,
          .shader = renderAPI.m_shaderManager.getDefaultShader(
              pain::DefaultShader::Texture),
      } //
  );
  renderAPI.m_materialManager.createMaterial(
      materialLine, //
      pain::MaterialCreationInfo{
          .color = pain::Colors::StrongPink,
          .shader = lineShader,
      } //
  );

  scene.createComponents(entity, pain::Transform2dComponent{},
                         pain::NativeScriptComponent{},
                         pain::MaterialComponent::create(backGround),
                         pain::SpriteComponent::create({
                             .layer = pain::RenderLayer::F,
                             .shape = pain::RectShape({2.f, 1.f}),
                         })); //
  //
  pain::Scene::emplaceScript<GraphRender>(entity, scene, camEntity, &font);
  return entity;
}

glm::vec2 GraphRender::screenToWorld(int x, int y) {
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

  const auto &[camCC, camTC] =
      getComponents<cmp::OrthoCamera, pain::Transform2dComponent>(m_camEntity);
  return camCC.screenToWorld(adjX, adjY, camTC);
}

void GraphRender::onEvent(const SDL_Event &event) {
  switch (event.type) {
  case SDL_MOUSEBUTTONDOWN: {
    if (event.button.button != SDL_BUTTON_LEFT)
      break;
    auto [sprite, transform] =
        getComponents<pain::SpriteComponent, pain::Transform2dComponent>();
    glm::vec2 mouse = screenToWorld(event.button.x, event.button.y);
    const pain::RectShape &rect = std::get<pain::RectShape>(sprite.m_shape);

    glm::vec2 half = rect.size * 0.5f;

    glm::vec2 min = transform.m_position - half;
    glm::vec2 max = transform.m_position + half;
    // if outside, break
    if (mouse.x < min.x || mouse.x > max.x || mouse.y < min.y ||
        mouse.y > max.y) {
      break;
    }
    m_dragOffset = mouse - transform.m_position;
    m_dragging = true;
    break;
  }

  case SDL_MOUSEBUTTONUP: {
    if (event.button.button == SDL_BUTTON_LEFT)
      m_dragging = false;
    const glm::vec2 &center =
        getComponent<pain::Transform2dComponent>().m_position;
    for (int i = 0; i < static_cast<int>(m_circles.size()); i++) {
      pain::Transform2dComponent &tc =
          getComponent<pain::Transform2dComponent>(m_circles[i]);
      tc.m_position -= m_centerCache;
      tc.m_position += center;
    }
    for (int i = 0; i < static_cast<int>(m_texts.size()); i++) {
      pain::Transform2dComponent &tc =
          getComponent<pain::Transform2dComponent>(m_texts[i]);
      tc.m_position -= m_centerCache;
      tc.m_position += center;
    }
    for (int i = 0; i < m_numEdges; i++) {
      auto [tc, sc] =
          getComponents<pain::Transform2dComponent, pain::SpriteComponent>(
              m_lines[i]);
      tc.m_position -= m_centerCache;
      tc.m_position += center;
      pain::LineShape &line = std::get<pain::LineShape>(sc.m_shape);
      line.destination -= m_centerCache;
      line.destination += center;
    }

    m_centerCache = center;
    break;
  }

  case SDL_MOUSEMOTION: {
    glm::vec2 mouse = screenToWorld(event.motion.x, event.motion.y);
    pain::Transform2dComponent &transform =
        getComponent<pain::Transform2dComponent>();

    if (m_dragging) {
      transform.m_position = mouse - m_dragOffset;
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
                         reg::Entity camEntity, pain::Font *font)
    : pain::WorldObject(entity, scene), m_font(font), m_camEntity(camEntity) {};

void GraphRender::generateGraph(pain::Scene &scene,
                                const std::vector<ConnectionGene> &links,
                                const std::vector<NodeGene> &neurons,
                                pain::Application &app) {
  std::vector<int> currentInput = {-1, -2, -3, -4, -5};
  // Couple of things to know to help create a beautiful graph:
  // 1. the genome will already be sorted from the first to last layer
  // 2. nodes will always have at least 1 link behind them, unless they are
  // the input
  // 2. nodes will always have at least 1 link after them, unless they are
  // the output
  // 3. inputs nodes don't have links behind
  // 4. output nodes don't have links forward

  struct LineGraphicInfo {
    int inNode;
    int outNode;
    const glm::vec2 from;
    const glm::vec2 to;
    float weight; // check if normalized [-1,1] or [0,1] later
  };

  // =====================================================
  // clear previous genome
  for (reg::Entity entity : m_circles) {
    scene.removeEntity(entity);
  }
  for (reg::Entity entity : m_lines) {
    scene.removeEntity(entity);
  }
  for (reg::Entity entity : m_texts) {
    scene.removeEntity(entity);
  }
  m_mapNodeEntity.clear();

  m_numNodes = 0;
  m_numEdges = 0;
  double maxWeight = -99999999999999;
  double minWeight = 99999999999999;
  std::vector<LineGraphicInfo> lineCoordMap;
  std::vector<Layer> layers;

  // =====================================================
  // Step 1: calculate outgoing and inDegree vectors
  std::map<int, std::vector<std::pair<int, double>>> outgoing;
  std::map<int, int> inDegree;
  for (const auto &link : links) {
    m_numEdges++;
    outgoing[link.m_InNodeId].push_back({link.m_OutNodeId, link.m_weight});
    inDegree[link.m_OutNodeId]++;
  }

  // Step 2: BFS
  while (!currentInput.empty()) {
    int size = static_cast<int>(layers.size());
    Layer &layer = layers.emplace_back(size, std::move(currentInput));
    std::vector<int> next;
    for (int node : layer.m_nodes) {
      for (auto [destinationNode, _] : outgoing[node]) {
        if (--inDegree[destinationNode] == 0) {
          next.push_back(destinationNode);
          m_numNodes++;
        }
      }
    }
    currentInput = std::move(next);
  }

  // Step 3: create layers (for drawing later)
  // map links from layer to layer. Graph is acyclical, i.e., no need to
  // test previous layers
  for (unsigned i = 0; i < layers.size(); i++) {
    const Layer &layer = layers[i];
    for (int node : layer.m_nodes) {
      for (unsigned j = i + 1; j < layers.size(); j++) {
        const Layer &olayer = layers[j];
        for (auto [destinationNode, weight] : outgoing[node]) {
          if (olayer.contains(destinationNode)) {
            maxWeight = std::max(maxWeight, weight);
            minWeight = std::min(minWeight, weight);
            lineCoordMap.emplace_back(node,                             //
                                      destinationNode,                  //
                                      layer.getCoord(node),             //
                                      olayer.getCoord(destinationNode), //
                                      weight                            //
            );
          }
        }
      }
    }
  }

  // =====================================================
  // Draw everything:
  m_circles.reserve(m_numNodes + currentInput.size());
  m_lines.reserve(m_numEdges);
  m_texts.reserve(currentInput.size());

  pain::MaterialManager &mm = app.getRenderApi().m_materialManager;
  // input text
  const Layer &inputLayer = layers[0];
  for (int node : inputLayer.m_nodes) {
    reg::Entity entity = scene.createEntity();
    scene.createComponents(
        entity, //
        pain::Transform2dComponent{inputLayer.getCoord(node) -
                                   glm::vec2(NODE_DIAMETER, NODE_DIAMETER / 2) +
                                   m_centerCache}, //
        pain::TextComponent{.text = "banana",
                            .scale = 8.f,
                            .align = pain::TextAlign::Right,
                            .font = *m_font} //
    );
    m_texts.push_back(entity);
  }

  for (const Layer &layer : layers) {
    // circles (nodes)
    for (int node : layer.m_nodes) {
      reg::Entity entity = scene.createEntity();
      scene.createComponents(
          entity,                                                           //
          pain::Transform2dComponent{layer.getCoord(node) + m_centerCache}, //
          pain::SpriteComponent::create(
              {.layer = pain::RenderLayer::F,
               .shape = pain::QuadShape{NODE_DIAMETER}}),         //
          pain::MaterialComponent{mm.getMaterial(materialNodes)}, //
          pain::ColorIndexComponent{pain::Colors::Black});        //
      m_circles.push_back(entity);
      m_mapNodeEntity.emplace(node, entity);
    }
  }
  // lines (edges)
  double maxAbsWeight = std::max(std::abs(maxWeight), std::abs(minWeight));
  for (const auto [inNode, outNode, orig, dest, weight] : lineCoordMap) {
    // float thickness =
    //     MIN_LINK_THICKNESS + std::abs(weight) / maxAbsWeight *
    //                              (MAX_LINK_THICKNESS - MIN_LINK_THICKNESS);
    double t = std::clamp(
        (weight - minWeight) / std::abs(maxWeight - minWeight), 0.0, 1.0);
    float thickness =
        t * (MAX_LINK_THICKNESS - MIN_LINK_THICKNESS) + MIN_LINK_THICKNESS;
    reg::Entity entity = scene.createEntity();
    scene.createComponents(
        entity,                                           //
        pain::Transform2dComponent{orig + m_centerCache}, //
        pain::SpriteComponent::create(
            {.layer = pain::RenderLayer::D,
             .shape = pain::LineShape{dest, thickness}}), //
        pain::MaterialComponent{mm.getMaterial(materialLine)},
        pain::ColorIndexComponent{pain::Colors::PastelGrey} //
    );                                                      //
    m_lines.push_back(entity);
  }
  m_maxWeight = maxWeight;
  m_minWeight = minWeight;
}

void GraphRender::updateWeights(
    const std::unordered_map<int, NodeInput> &weights,
    const std::vector<double> &inputs) {
  for (const auto [node, entity] : m_mapNodeEntity) {
    pain::Color &color = getComponent<pain::ColorIndexComponent>(entity).color;
    double weight = weights.at(node).outputValue;
    double t = std::clamp(
        (weight - m_minWeight) / std::abs(m_maxWeight - m_minWeight), 0.0, 1.0);
    int index = static_cast<int>(t * (m_palette.size() - 1));
    color = m_palette[index];
    color.value =
        (color.value & 0x00FFFFFF) | (static_cast<uint8_t>(t * 255) << 24);
  }
}

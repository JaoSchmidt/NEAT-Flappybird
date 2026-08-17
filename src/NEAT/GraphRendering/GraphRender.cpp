#include "NEAT/GraphRendering/GraphRender.h"
#include "Assets/ManagerTexture.h"
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
constexpr std::string_view materialFrame = "GraphFrame";
constexpr std::string_view materialBackground = "GraphBackground";

// score, deaths, generation, species

GraphRender::Layer::Layer(int layer, std::vector<int> nodes)
    : m_nodes(std::move(nodes))
{
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
                                reg::Entity camEntity)
{
  const float zoom = 1.f;
  const glm::vec2 center{0.06f, -1.71f};

  reg::Entity entity = scene.createEntity("GraphRenderRoot");
  // pain::Shader &nodeShader = renderAPI.m_shaderManager.getDefaultShader(
  //     pain::DefaultShader::Circles);
  // pain::Shader &backGroundShader =
  //     renderAPI.m_shaderManager.getDefaultShader(pain::DefaultShader::Texture);
  pain::Shader &frameShader = renderAPI.m_shaderManager.loadShaderFromFile(
      "GraphFrame", "resources/shaders/graphFrame.glsl");
  pain::Shader &nodeShader = renderAPI.m_shaderManager.loadShaderFromFile(
      "GraphNodeShader", "resources/shaders/graphNodes.glsl");
  pain::Shader &lineShader =
      renderAPI.m_shaderManager.getDefaultShader(pain::DefaultShader::Texture);

  // pain::Shader &lineShader = renderAPI.m_shaderManager.loadShaderFromFile(
  //     "LineGraphShader", "resources/shaders/graphLine.glsl");
  renderAPI.m_materialManager.createMaterial(
      materialNodes, //
      pain::MaterialCreationInfo{
          .color = pain::Colors::FullWhite,
          .shader = nodeShader,
      } //
  );
  renderAPI.m_materialManager.createMaterial(
      materialBackground, //
      pain::MaterialCreationInfo{
          .color = pain::Color::fromRGB(0x444444),
          .shader = renderAPI.m_shaderManager.getDefaultShader(
              pain::DefaultShader::Texture) //
      } //
  );
  renderAPI.m_materialManager.createMaterial(
      materialFrame, //
      pain::MaterialCreationInfo{
          .color = pain::Color::fromRGB(0xdd8e40),
          .shader = frameShader //
      } //
  );
  renderAPI.m_materialManager.createMaterial(
      materialLine, //
      pain::MaterialCreationInfo{
          .color = pain::Colors::StrongPink,
          .shader = lineShader,
      } //
  );

  scene.createComponents(entity, cmp::Pos2d{center}, cmp::Script{} //
  );                                                               //
  //
  pain::Scene::emplaceScript<GraphRender>(
      entity, scene, camEntity,
      &renderAPI.m_fontManager.createFont(
          "SourceSans", "resources/default/fonts/SourceSans3-Regular.ttf", 60),
      center);
  return entity;
}

glm::vec2 GraphRender::screenToWorld(int x, int y)
{
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
      getComponents<cmp::Cam2d, cmp::Pos2d>(m_camEntity);
  return camCC.screenToWorld(adjX, adjY, camTC);
}

void GraphRender::onEvent(const SDL_Event &event)
{
  switch (event.type) {
  case SDL_MOUSEBUTTONDOWN: {
    if (event.button.button != SDL_BUTTON_LEFT)
      break;
    cmp::Pos2d &transform = getComponent<cmp::Pos2d>();
    glm::vec2 mouse = screenToWorld(event.button.x, event.button.y);

    glm::vec2 half = glm::vec2(0.5f);

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
    const glm::vec2 &center = getComponent<cmp::Pos2d>().m_position;
    for (int i = 0; i < static_cast<int>(m_circles.size()); i++) {
      cmp::Pos2d &tc = getComponent<cmp::Pos2d>(m_circles[i]);
      tc.m_position -= m_centerCache;
      tc.m_position += center;
    }
    for (int i = 0; i < static_cast<int>(m_misc.size()); i++) {
      cmp::Pos2d &tc = getComponent<cmp::Pos2d>(m_misc[i]);
      tc.m_position -= m_centerCache;
      tc.m_position += center;
    }
    for (int i = 0; i < m_numEdges; i++) {
      auto [tc, sc] = getComponents<cmp::Pos2d, cmp::Sprite>(m_lines[i]);
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
    cmp::Pos2d &transform = getComponent<cmp::Pos2d>();

    if (m_dragging) {
      transform.m_position = mouse - m_dragOffset;
    }
    break;
  }

  default:
    break;
  }
}

void populateMisc(std::vector<reg::Entity> &misc, pain::Scene &scene,
                  pain::Application &app, const glm::vec2 &center = {0, 0})
{
  pain::MaterialManager &mm = app.getRenderApi().m_materialManager;

  reg::Entity entity = scene.createEntity("Graph Frame");
  scene.createComponents(entity, cmp::Pos2d{center}, cmp::Script{},
                         cmp::Material::create(mm.getMaterial(materialFrame)),
                         cmp::Sprite::create({
                             .layer = pain::RenderLayer::C,
                             .shape = pain::RectShape({4.f, 1.0f}),
                         })); //
  misc.push_back(entity);
  scene.createComponents(
      entity, cmp::Pos2d{center}, cmp::Script{},
      cmp::Material::create(mm.getMaterial(materialBackground)),
      cmp::Sprite::create({
          .layer = pain::RenderLayer::B,
          .shape = pain::RectShape({6.f, 1.2f}),
      })); //
  misc.push_back(entity);
}

GraphRender::GraphRender(reg::Entity entity, pain::Scene &scene,
                         reg::Entity camEntity, pain::Font *font,
                         const glm::vec2 &center)
    : pain::WorldObject(entity, scene), m_centerCache(center), m_font(font),
      m_camEntity(camEntity) {};

void GraphRender::generateGraph(pain::Scene &scene,
                                const std::vector<ConnectionGene> &links,
                                const std::vector<std::string> &inputNames,
                                pain::Application &app)
{
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
  for (reg::Entity entity : m_misc) {
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
  m_misc.reserve(currentInput.size() // input text
                 + 1                 // frame
                 + 1                 // background
  );

  pain::MaterialManager &mm = app.getRenderApi().m_materialManager;
  populateMisc(m_misc, scene, app, m_centerCache);
  // input text
  const Layer &inputLayer = layers[0];
  for (int node : inputLayer.m_nodes) {
    if (node < 0) {
      reg::Entity entity = scene.createEntity("GraphText");
      scene.createComponents(
          entity, //
          cmp::Pos2d{inputLayer.getCoord(node) -
                     glm::vec2(NODE_DIAMETER, NODE_DIAMETER / 2) +
                     m_centerCache}, //
          cmp::Text{.text = inputNames[-node - 1],
                    .scale = 6.f,
                    .align = pain::TextAlign::Right,
                    .font = *m_font} //
      );
      m_misc.push_back(entity);
    }
  }

  for (const Layer &layer : layers) {
    // circles (nodes)
    for (int node : layer.m_nodes) {
      reg::Entity entity = scene.createEntity("Graph Node");
      scene.createComponents(
          entity,                                           //
          cmp::Pos2d{layer.getCoord(node) + m_centerCache}, //
          cmp::Sprite::create({.layer = pain::RenderLayer::F,
                               .shape = pain::QuadShape{NODE_DIAMETER}}), //
          cmp::Material{mm.getMaterial(materialNodes)},                   //
          cmp::ColorIdx{pain::Colors::Black});                            //
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
    reg::Entity entity = scene.createEntity("GraphEdge");
    PLOG_I("Line coords: ({},{}) -> ({},{})", TP_VEC2(orig), TP_VEC2(dest));

    scene.createComponents(
        entity,                           //
        cmp::Pos2d{orig + m_centerCache}, //
        cmp::Sprite::create(
            {.layer = pain::RenderLayer::D,
             .shape = pain::LineShape{dest + m_centerCache, thickness}}), //
        cmp::Material{mm.getMaterial(materialLine)},
        cmp::ColorIdx{pain::Colors::PastelGrey} //
    );                                          //
    m_lines.push_back(entity);
  }
  m_maxWeight = maxWeight;
  m_minWeight = minWeight;
}

void GraphRender::updateWeights(
    const std::unordered_map<int, NodeInput> &weights,
    const std::vector<double> &inputs)
{
  for (const auto [node, entity] : m_mapNodeEntity) {
    pain::Color &color = getComponent<cmp::ColorIdx>(entity).color;
    double weight = weights.at(node).outputValue;
    double t = std::clamp(
        (weight - m_minWeight) / std::abs(m_maxWeight - m_minWeight), 0.0, 1.0);
    int index = static_cast<int>(t * (m_palette.size() - 1));
    color = m_palette[index];
    color.value =
        (color.value & 0x00FFFFFF) | (static_cast<uint8_t>(t * 255) << 24);
  }
}

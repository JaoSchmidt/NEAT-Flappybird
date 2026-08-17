#include "FlappyGame.h"
#include "Assets/ManagerTexture.h"
#include <cstdlib>

std::tuple<PlayerController *, pain::Material &,
           std::vector<ObstaclesController *>>
FlappyGame::createHelper(pain::Scene &scene, pain::Application &app)
{
  pain::RenderApi &renderAPI = app.getRenderApi();
  pain::Shader &obstacleShader = renderAPI.m_shaderManager.getDefaultShader(
      pain::DefaultShader::SimpleTriangles);
  pain::Shader &defaultShader =
      renderAPI.m_shaderManager.getDefaultShader(pain::DefaultShader::Texture);

  pain::Texture &playerTexture =
      pain::TextureManager::createTexture("resources/textures/Player.png");

  pain::Material &playerMaterial = renderAPI.m_materialManager.createMaterial(
      "Player Material", //
      pain::MaterialCreationInfo{
          .color = pain::Colors::SkyBlue,
          .params = std::monostate{},
          .shader = defaultShader,
          .texture = playerTexture,
      } //
  );

  pain::Material &obstacleMaterial = renderAPI.m_materialManager.createMaterial(
      "Obstacle Material", //
      pain::MaterialCreationInfo{
          .color = pain::Colors::SkyBlue,
          .params = std::monostate{},
          .shader = obstacleShader,
      } //
  );
  reg::Entity player = createPlayer(scene, playerMaterial);
  PlayerController *pc = &scene.getNativeScript<PlayerController>(player);

  std::vector<ObstaclesController *> obstacles;
  obstacles.reserve(s_numberOfObstacles);
  for (char i = 0; i < s_numberOfObstacles; i++) {
    reg::Entity e = ObstaclesController::create(scene, obstacleMaterial);
    ObstaclesController &oc = scene.getNativeScript<ObstaclesController>(e);
    obstacles.emplace_back(&oc);
  };
  return {pc, obstacleMaterial, std::move(obstacles)};
}

reg::Entity FlappyGame::create(pain::Scene &scene, pain::Application &app)
{
  const int w = 1024;
  const int h = 768;

  pain::Dummy2dCamera::createStaticCamera(scene, w, h, 1.f);

  auto [pc, obstacleMaterial, obstacles] = createHelper(scene, app);
  pain::Scene::emplaceScript<FlappyGame>(scene.getEntity(), scene, pc,
                                         obstacleMaterial, std::move(obstacles),
                                         app);
  return scene.getEntity();
}
FlappyGame::FlappyGame(reg::Entity entity, pain::Scene &scene,
                       PlayerController *pc, pain::Material &om,
                       std::vector<ObstaclesController *> obc,
                       pain::Application &a)
    : pain::WorldObject(entity, scene), m_playerController(pc),
      m_obstacles(std::move(obc)), m_obstaclesMaterial(om), m_app(a) {};

void FlappyGame::changeObstaclesColors(pain::Color color)
{
  m_obstaclesMaterial.m_color = color;
}

void FlappyGame::onCreate()
{

  m_panelID = painless::customPanel::addToPanel(
      "Controller",
      [this]() { //
        ImGui::Text("Obstacles Parameters Settings");
        ImGui::Text("Number of Obstacles: %d", s_numberOfObstacles);
        ImGui::InputFloat("Obstacles Spacing", &m_obstaclesSpacing, 0.01F, 1.0F,
                          "%.3f");
        ImGui::InputFloat("Max Interval", &m_maxInterval, 0.1F, 1.0F, "%.3f");
        ImGui::InputFloat("Interval Time", &m_intervalTime, 0.1F, 1.0F, "%.3f");
        ImGui::InputFloat("Obstacle Speed", &m_defaultObstacleSpeed, 0.01F,
                          1.0F, "%.3F");
        ImGui::InputFloat("Color Interval", &m_colorInterval, 0.1F, 1.0F,
                          "%.3f");
        ImGui::InputFloat("Height Interval", &m_heightInterval, 0.1F, 1.0F,
                          "%.3f");
        ImGui::SeparatorText("Info");
        ImGui::Text("Obstacle Spawn counter:% .2F seconds",
                    m_obstaclesInterval);
        ImGui::Text(" Last Obstacle index : %.2d ", m_index);
        ImGui::Text(" Points : %.4d ", m_points);
        ImGui::Text(" Loses : %.4d ", m_loses);

        double time = m_app.getTimeMultiplier();
        ImGui::InputDouble("Time Multiplier ", &time, 100., 1.0, "%.3f");
        m_app.setTimeMultiplier(time);

        if (ImGui::Button("Toogle Rendering")) {
          m_rendering = !m_rendering;
          m_app.setRendereing(m_rendering);
        }
        ImGui::Text("Rendering is %s", m_rendering ? "ON" : "OFF");
      },
      2);
}

void FlappyGame::onRender(pain::RenderContext &_, pain::DeltaTime currentTime)
{

  m_waveColor =
      m_waveColor + fmod(m_colorInterval * currentTime.getSecondsf(), 360.F);

  const auto waveColorRadians = glm::radians(m_waveColor);
  // change obstacle color
  pain::Color color(125 + sin(waveColorRadians) * 124,               // red
                    76.5 + sin(waveColorRadians + M_PI / 4) * 76.5,  // green
                    102 + sin(waveColorRadians + M_PI * 3 / 4) * 102 // blue
  );
  m_obstaclesMaterial.m_color = color;
}

void FlappyGame::onUpdate(pain::DeltaTime deltaTime)
{
  if (m_isRunning) {
    // Overall game
    // 1. if obstacle is outside camera, call onDestroy
    // 2. check if player hits obstacles
    // 3. if hits, remove one life
    // 4. if 0 lifes, score menu

    // spawn obstacles
    m_obstaclesInterval -= m_intervalTime * deltaTime.getSecondsf();
    if (m_obstaclesInterval <= 0) {
      m_obstaclesInterval = m_maxInterval;
      const float randAngle =
          static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * M_PI * 2;

      m_index = (m_index + 1) % s_numberOfObstacles;
      reviveObstacle(m_index, randAngle, true);
      m_index = (m_index + 1) % s_numberOfObstacles;
      reviveObstacle(m_index, randAngle, false);
    }

    for (char i = 0; i < s_numberOfObstacles; i++) {
      if (checkIntersection(*m_obstacles[i]))
        afterLosing();
    }
  }
}

void FlappyGame::afterLosing()
{
  m_loses++;
  m_points = 0;
  // reset Player position
  m_playerController->resetPosition();
  // clear obstacles
  for (char i = 0; i < s_numberOfObstacles; i++)
    m_obstacles[i]->revive(0, 0, false, &m_points);
}

void FlappyGame::reviveObstacle(int index, float randomAngle, bool upsideDown)
{
  const float height =
      upsideDown ? sin(randomAngle) * 0.7F + 0.75F + m_obstaclesSpacing
                 : sin(randomAngle) * 0.7F - 1.25F;
  m_obstacles.at(index)->revive(m_defaultObstacleSpeed, height, upsideDown,
                                &m_points);
}

template <std::size_t T>
glm::vec2 FlappyGame::projection(const std::array<glm::vec2, T> &shape,
                                 const glm::vec2 &axis)
{
  float min = glm::dot(shape[0], axis);
  float max = min;
  for (size_t i = 1; i < shape.size(); i++) {
    float projection = glm::dot(shape[i], axis);
    min = std::min(min, projection);
    max = std::max(max, projection);
  }
  return {min, max};
}

bool FlappyGame::checkIntersection(const ObstaclesController &obstacle)
{
  auto &ptc = m_playerController->getComponent<cmp::Pos2d>();
  auto &prc = m_playerController->getComponent<cmp::Rot>();
  auto &psc = m_playerController->getComponent<cmp::Sprite>();
  auto &otc = obstacle.getComponent<cmp::Pos2d>();
  auto &osc = obstacle.getComponent<cmp::Sprite>();

  // get quad vertices
  constexpr glm::vec4 quadVertexPositions[4] = {
      glm::vec4(-0.5f, -0.5f, 0.f, 1.f),
      glm::vec4(0.5f, -0.5f, 0.f, 1.f),
      glm::vec4(0.5f, 0.5f, 0.f, 1.f),
      glm::vec4(-0.5f, 0.5f, 0.f, 1.f),
  };

  const pain::RectShape &qs = std::get<pain::RectShape>(psc.m_shape);
  const glm::mat4 transform = pain::Renderer2d::getTransform(
      ptc.m_position, qs.size, prc.m_rotationRadians);

  std::array<glm::vec2, 4> qVertices = {
      transform * quadVertexPositions[0],
      transform * quadVertexPositions[1],
      transform * quadVertexPositions[2],
      transform * quadVertexPositions[3],
  };

  // triangle
  constexpr glm::vec4 triVertexPositions[3] = {
      glm::vec4(0.0f, 0.5f, 0.f, 1.f),
      glm::vec4(0.5f, -0.5f, 0.f, 1.f),
      glm::vec4(-0.5f, -0.5f, 0.f, 1.f),
  };
  const pain::TriangleShape &ts = std::get<pain::TriangleShape>(osc.m_shape);
  const glm::mat4 transformTri =
      pain::Renderer2d::getTransform(otc.m_position, {ts.base, ts.height});
  const std::array<glm::vec2, 3> tVertices = {
      transformTri * triVertexPositions[0], //
      transformTri * triVertexPositions[1], //
      transformTri * triVertexPositions[2], //
  };

  std::vector<glm::vec2> axes;
  for (size_t i = 0; i < 4; i++) {
    glm::vec2 edge = qVertices[(i + 1) % 4] - qVertices[i];
    glm::vec2 axis(-edge.y, edge.x); // Perpendicular to the edge
    axis = glm::normalize(axis);
    axes.push_back(axis);
  }
  for (size_t i = 0; i < 3; i++) {
    glm::vec2 edge = tVertices[(i + 1) % 3] - tVertices[i];
    glm::vec2 axis(-edge.y, edge.x);
    axis = glm::normalize(axis);
    axes.push_back(axis);
  }
  // Perform SAT check on all axes
  for (const glm::vec2 &axis : axes) {
    auto boundA = projection(qVertices, axis);
    auto boundB = projection(tVertices, axis);

    // Check for overlap
    if (boundA.y < boundB.x || boundB.y < boundA.x)
      return false; // No collision
  }

  return true;
}

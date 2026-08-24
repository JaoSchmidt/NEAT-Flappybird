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
      m_obstaclesMaterial(om), m_app(a)
{
  m_playerController->m_obstacles = std::move(obc);
};

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

    auto closest = m_playerController->getClosestObstacles();
    if (checkIfLost(closest.up))
      return;
    if (checkIfLost(closest.down))
      return;
  }
}
bool FlappyGame::checkIfLost(int obstacleId)
{
  if (obstacleId >= 0) {
    ObstaclesController &obstacle =
        *m_playerController->m_obstacles.at(obstacleId);
    float x = obstacle.getComponent<cmp::Pos2d>().m_position.x;
    if (x < -0.2F && m_playerController->checkIntersection(obstacle)) {
      afterLosing();
      return true;
    }
  }
  return false;
}
void FlappyGame::afterLosing()
{
  m_loses++;
  m_points = 0;
  // reset Player position
  m_playerController->resetPosition();
  // clear obstacles
  for (char i = 0; i < s_numberOfObstacles; i++)
    m_playerController->m_obstacles[i]->revive(0, 0, false, &m_points);
}

void FlappyGame::reviveObstacle(int index, float randomAngle, bool upsideDown)
{
  const float height =
      upsideDown ? sin(randomAngle) * 0.7F + 0.75F + m_obstaclesSpacing
                 : sin(randomAngle) * 0.7F - 1.25F;
  m_playerController->m_obstacles.at(index)->revive(
      m_defaultObstacleSpeed, height, upsideDown, &m_points);
}

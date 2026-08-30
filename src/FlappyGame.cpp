#include "FlappyGame.h"
#include "Assets/ManagerTexture.h"
#include <cstdlib>

std::tuple<std::vector<PlayerController *>, pain::Material &,
           std::vector<ObstaclesController *>>
FlappyGame::createHelper(pain::Scene &scene, pain::Application &app,
                         int numPlayers)
{
  pain::RenderApi &renderAPI = app.getRenderApi();
  pain::Shader &obstacleShader = renderAPI.m_shaderManager.getDefaultShader(
      pain::DefaultShader::SimpleTriangles);
  pain::Material &obstacleMaterial = renderAPI.m_materialManager.createMaterial(
      "Obstacle Material", //
      pain::MaterialCreationInfo{
          .color = pain::Colors::FullWhite,
          .params = std::monostate{},
          .shader = obstacleShader,
      } //
  );
  std::vector<PlayerController *> players;
  players.reserve(numPlayers);
  for (int i = 0; i < numPlayers; i++) {
    reg::Entity player = createPlayer(scene, renderAPI);
    players.push_back(&scene.getNativeScript<PlayerController>(player));
  }

  std::vector<ObstaclesController *> obstacles;
  obstacles.reserve(s_numberOfObstacles);
  for (char i = 0; i < s_numberOfObstacles; i++) {
    reg::Entity e = ObstaclesController::create(scene, obstacleMaterial);
    ObstaclesController &oc = scene.getNativeScript<ObstaclesController>(e);
    obstacles.emplace_back(&oc);
  };
  return {std::move(players), obstacleMaterial, std::move(obstacles)};
}

reg::Entity FlappyGame::create(pain::Scene &scene, pain::Application &app)
{
  const int w = 1024;
  const int h = 768;

  pain::Dummy2dCamera::createStaticCamera(scene, w, h, 1.f);

  auto [pcs, obstacleMaterial, obstacles] = createHelper(scene, app);
  pain::Scene::emplaceScript<FlappyGame>(scene.getEntity(), scene, pcs[0],
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
        ImGui::Text(" Last Obstacle index : %.2d ", m_recentObstacleIndex);
        ImGui::Text(" Score : %.4d ", m_gameScore);
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

      m_recentObstacleIndex = (m_recentObstacleIndex + 1) % s_numberOfObstacles;
      reviveObstacle(m_recentObstacleIndex, randAngle, true);
      m_recentObstacleIndex = (m_recentObstacleIndex + 1) % s_numberOfObstacles;
      reviveObstacle(m_recentObstacleIndex, randAngle, false);
    }

    auto &visible = m_playerController->getVisibleObstacles();
    for (ObstaclesController *obs : visible) {
      if (checkIfLost(obs))
        return;
    }
  }
}
bool FlappyGame::checkIfLost(ObstaclesController *obstacle)
{
  if (obstacle) {
    float x = obstacle->getComponent<cmp::Pos2d>().m_position.x;
    if (x < -0.2F && m_playerController->checkIntersection(*obstacle)) {
      afterLosing();
      return true;
    }
  }
  return false;
}
void FlappyGame::afterLosing()
{
  m_loses++;
  m_gameScore = 0;
  // reset Player position
  m_playerController->resetPosition();
  // clear obstacles
  for (char i = 0; i < s_numberOfObstacles; i++)
    m_playerController->m_obstacles[i]->revive(0, 0, false, &m_gameScore);
}

void FlappyGame::reviveObstacle(int index, float randomAngle, bool upsideDown)
{
  const float height =
      upsideDown ? sin(randomAngle) * 0.7F + 0.75F + m_obstaclesSpacing
                 : sin(randomAngle) * 0.7F - 1.25F;
  m_playerController->m_obstacles.at(index)->revive(
      m_defaultObstacleSpeed, height, upsideDown, &m_gameScore);
}

#include <pain.h>
#include <painless.h>

#include "NEAT/Population.h"
#include <glm/ext/matrix_transform.hpp>
#include <glm/fwd.hpp>

pain::Application *pain::createApplication() {
  // Retrieve the context the player will alter when using the launcher
  IniConfig ini;
  ini.readAndUpdate();

  // Retrieve the app context defined inside "resources/InternalConfig.ini"
  InternalConfig internalIni;
  internalIni.readAndUpdate(ini.assetsPath.value);

  // Create the application + OpenGL + Event contexts
  Application *app = Application::createApplication( //
      {.title = internalIni.title.get().c_str(),     //
       .defaultWidth = ini.defaultWidth.get(),       //
       .defaultHeight = ini.defaultHeight.get(),
       .defaultZoom2d = internalIni.zoomLevel.get(),
       .fullWindow = ini.fullwindow.get(),
       .fullScreen = ini.fullscreen.get()},                  //
      {.swapChainTarget = internalIni.swapChainTarget.get()} //
  );

  // Create the ECS World Scene
  // pain::Scene& scene = app->createWorldSceneComponents(
  //     internalIni.gridSize.get(), pain::NativeScriptComponent{},
  //     pain::LuaScriptComponent(scene.getEntity()));

  app->getRenderers().m_renderer2d.setCellGridSize(internalIni.gridSize.get());

  pain::Scene &scene = app->getWorldScene();
  scene.createComponents(scene.getEntity(), pain::NativeScriptComponent{},
                         pain::LuaScriptComponent::create(scene.getEntity()));

  pain::BasicScene::syncSystems(scene);

  // (Optional) Creating the ECS UI scene
  pain::UIScene &uiScene = app->createUIScene();
  uiScene.addSystem<pain::Systems::ImGuiSys>(app->getRenderContext(),
                                             app->getRenderWindow());
  uiScene.createComponents(uiScene.getEntity(), painless::ImGuiComponent{});

  // (Optional) A small native script that works as our game engine editor
  painless::Editor &editor = painless::Editor::create(uiScene, *app);

  // (Optional) Define a small native script for the world scene
  // that will be executed on as root script. Must have added
  // System::NativeScript
  FlappyGame::create( //
      scene,          //
      *app,
      editor //
  );
  return app;
}

#ifdef PLATFORM_IS_LINUX
int main(int argc, char *argv[]) {
  UNUSED(argc)
  UNUSED(argv)
#elif defined PLATFORM_IS_WINDOWS
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR lpCmdLine,
                   int nCmdShow) {
#else
int main() {
#endif
  bool isSettingsGuiNeeded = pain::Pain::initiateIni();
  EndGameFlags flags;
  flags.restartGame = !isSettingsGuiNeeded;
  if (isSettingsGuiNeeded) {
    pain::Application *app = painless::createLauncher();
    flags = pain::Pain::runAndDeleteApplication(app);
  }
  while (flags.restartGame) {
    pain::Application *app = pain::createApplication();
    flags = pain::Pain::runAndDeleteApplication(app);
  }
  return 0;
}

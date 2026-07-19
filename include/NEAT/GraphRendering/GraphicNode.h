#include <pain.h>
#include <vector>

struct GraphicLine;

struct GraphicNode : public pain::WorldObject {
  using pain::WorldObject::WorldObject;
  void turnUp();
  static float maxGraphWeight;
  static float minGraphWeight;
};

struct GraphicLine : public pain::WorldObject {
  using pain::WorldObject::WorldObject;
  void turnUp();
  static float maxGraphWeight;
  static float minGraphWeight;
};

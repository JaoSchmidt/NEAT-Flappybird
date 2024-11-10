#pragma once
#include "Core.h"
#include "NEAT/Activation.h"

#include <cmath>
#include <pain.h>
#include <vector>

class Individual;

struct InnovationStatic {
  int m_innovation;
  int m_InNodeId;
  int m_OutNodeId;

  InnovationStatic(int innovation, int inNodeId, int outNodeId)
      : m_innovation(innovation), m_InNodeId(inNodeId), m_OutNodeId(outNodeId)
  {
  }
  InnovationStatic(InnovationStatic &&o) = default;
  InnovationStatic &operator=(InnovationStatic &&o) = default;

  COPIES(InnovationStatic);
};

struct ConnectionGene {
  int m_InNodeId;
  int m_OutNodeId;
  int m_innovation;
  bool m_enable;
  double m_weight;

  ConnectionGene(int inputNodeId, int outputNodeId, double weight, bool enable,
                 int innovation)
      : m_InNodeId{inputNodeId}, m_OutNodeId{outputNodeId},
        m_innovation{innovation}, m_enable{enable}, m_weight{weight}
  {
  }

  ConnectionGene(ConnectionGene &&o) = default;
  ConnectionGene &operator=(ConnectionGene &&o) = default;

  ConnectionGene clone() const
  {
    return ConnectionGene(m_InNodeId, m_OutNodeId, m_weight, m_enable,
                          m_innovation);
  }

  COPIES(ConnectionGene)
};

// represents a neuron
struct NodeGene {
  int m_neuron_id;
  double m_bias;
  ActivationFunction m_activationFunction;

  NodeGene(int nodeId, double bias) : m_neuron_id{nodeId}, m_bias{bias} {}
  NodeGene(NodeGene &&o) = default;
  NodeGene &operator=(NodeGene &&o) = default;
  COPIES(NodeGene)
  // prefered way to copy
  NodeGene clone() const { return NodeGene(m_neuron_id, m_bias); };
};

// all genes
struct Genome {
  std::vector<NodeGene> m_neurons;
  std::vector<ConnectionGene> m_links;
  std::vector<InnovationStatic> &m_globalInnovations;

  void addConnectionAndSort(ConnectionGene connGene, int numInputs);
  void addNode(ConnectionGene &oldGene, int numInputs, int numOutputs);
  void removeConnection(ConnectionGene &connGene, int numInputs);
  void removeNode(NodeGene &nodeGene, int numInputs);
  // Non Structural Mutations
  void setBias(int neuron_id, double bias);
  void setWeight(int link_id, double weight);
  const std::vector<double> run(const std::vector<double> &inputs,
                                int numInputs, int numOutputs) const;

private:
  friend class Individual;
  const bool willIsolateOutNode(const ConnectionGene &link) const;
  const bool willIsolateInNode(const ConnectionGene &link) const;
  bool hasPathDFS(int startNode, int targetNode) const;
  void topologySortNN(int numInputs);
  // get correct innovation or create new one
  int loadInnovation(int inputNodeId, int outputNodeId);

public:
  ~Genome() = default;
  Genome(std::vector<InnovationStatic> &globalInnovations)
      : m_neurons{}, m_links{}, m_globalInnovations{globalInnovations}
  {
  }

  Genome(std::vector<NodeGene> neurons, std::vector<ConnectionGene> links,
         std::vector<int> layers,
         std::vector<InnovationStatic> &globalInnovations)
      : m_neurons(std::move(neurons)), m_links(std::move(links)),
        m_globalInnovations(globalInnovations)
  {
  }

  // Copy Constructor
  Genome(const Genome &other)
      : m_neurons(other.m_neurons), m_links(other.m_links),
        m_globalInnovations(other.m_globalInnovations)
  {
  }
  Genome &operator=(const Genome &other)
  {
    if (this != &other) {
      m_neurons = other.m_neurons;
      m_links = other.m_links;
      m_globalInnovations = other.m_globalInnovations;
    }
    LOG_I("globalInnovations {}", m_globalInnovations.size());
    return *this;
  }
  Genome(Genome &&other) noexcept
      : m_neurons(std::move(other.m_neurons)),
        m_links(std::move(other.m_links)),
        m_globalInnovations(other.m_globalInnovations)
  {
  }
  Genome &operator=(Genome &&other) noexcept
  {
    if (this != &other) {
      m_neurons = std::move(other.m_neurons);
      m_links = std::move(other.m_links);
      m_globalInnovations = other.m_globalInnovations;
    }
    return *this;
  }
};

#include "NEAT/Individuals.h"
#include <cstddef>

Individual &Individual::crossover(const Individual &other) const
{
  // Determine the fitter parent
  const Individual &fitterParent =
      (m_fitness >= other.m_fitness) ? *this : other;
  const Individual &lessFitParent =
      (m_fitness < other.m_fitness) ? *this : other;

  // Create a new genome for the offspring
  Genome offspringGenome;

  // Crossover nodes with matching neuron IDs
  for (const auto &node : fitterParent.m_genome.m_neurons) {
    auto matchIt = std::find_if(
        lessFitParent.m_genome.m_neurons.begin(),
        lessFitParent.m_genome.m_neurons.end(), [&](const NodeGene &otherNode) {
          return otherNode.m_neuron_id == node.m_neuron_id;
        });

    if (matchIt != lessFitParent.m_genome.m_neurons.end()) {
      // Node with matching ID found, choose biases randomly from parents
      double chosenBias = (m_config.m_rng.uniform<double>() < 0.5)
                              ? node.m_bias
                              : matchIt->m_bias;
      ActivationFunction chosenActivationFunction =
          (m_config.m_rng.uniform<double>() < 0.5)
              ? node.m_activationFunction
              : matchIt->m_activationFunction;
      offspringGenome.m_neurons.emplace_back(node.m_neuron_id, chosenBias,
                                             node.m_layer_id);
    } else {
      // No matching node, inherit the node from the fitter parent
      offspringGenome.m_neurons.push_back(node.clone());
    }
  }

  // Crossover links with matching innovations
  for (const auto &link : fitterParent.m_genome.m_links) {
    auto matchIt =
        std::find_if(lessFitParent.m_genome.m_links.begin(),
                     lessFitParent.m_genome.m_links.end(),
                     [&](const ConnectionGene &otherLink) {
                       return otherLink.m_innovation == link.m_innovation;
                     });

    if (matchIt != lessFitParent.m_genome.m_links.end()) {
      // Matching link found, randomly choose weight from either parent
      double chosenWeight = (m_config.m_rng.uniform<double>() < 0.5)
                                ? link.m_weight
                                : matchIt->m_weight;
      offspringGenome.m_links.emplace_back(link.m_InNodeId, link.m_OutNodeId,
                                           chosenWeight, link.m_enable,
                                           link.m_innovation);
    } else {
      // No matching link, inherit from the fitter parent
      offspringGenome.m_links.push_back(link.clone());
    }
  }

  // Add excess genes from the fitter parent
  for (const auto &excessNode : fitterParent.m_genome.m_neurons) {
    auto matchIt = std::find_if(
        offspringGenome.m_neurons.begin(), offspringGenome.m_neurons.end(),
        [&](const NodeGene &node) {
          return node.m_neuron_id == excessNode.m_neuron_id;
        });

    if (matchIt == offspringGenome.m_neurons.end()) {
      offspringGenome.m_neurons.push_back(excessNode.clone());
    }
  }

  for (const auto &excessLink : fitterParent.m_genome.m_links) {
    auto matchIt = std::find_if(
        offspringGenome.m_links.begin(), offspringGenome.m_links.end(),
        [&](const ConnectionGene &link) {
          return link.m_innovation == excessLink.m_innovation;
        });

    if (matchIt == offspringGenome.m_links.end()) {
      offspringGenome.m_links.push_back(excessLink.clone());
    }
  }

  // Create the offspring Individual with a new genome and reset fitness
  Individual *offspring =
      new Individual(std::move(offspringGenome), 0.0, m_config);

  return *offspring;
}

double Individual::calculateDelta(const Individual &other) const
{
  // Coefficients from the configuration
  double c1 = m_config.m_c1; // Assuming this is where c1 is defined
  double c2 = m_config.m_c2; // Example value for c2, adjust as necessary
  double c3 = m_config.m_c3; // Example value for c3, adjust as necessary

  int excessCount = 0;
  int disjointCount = 0;
  double totalWeightDifference = 0.0;
  int matchingCount = 0;

  // Count matching, excess, and disjoint genes
  for (const auto &link : m_genome.m_links) {
    auto otherLinkIt = std::find_if(
        other.m_genome.m_links.begin(), other.m_genome.m_links.end(),
        [&link](const ConnectionGene &otherLink) {
          return otherLink.m_innovation == link.m_innovation;
        });
    if (otherLinkIt != other.m_genome.m_links.end()) {
      // Matching gene
      double weightDifference = std::abs(link.m_weight - otherLinkIt->m_weight);
      totalWeightDifference += weightDifference; // Sum the differences
      matchingCount++;
    }
  }

  // Count excess and disjoint genes
  excessCount = (m_genome.m_links.size() - matchingCount);
  disjointCount = (other.m_genome.m_links.size() - matchingCount);

  // Calculate W, the average weight difference of matching genes
  double W =
      (matchingCount > 0) ? (totalWeightDifference / matchingCount) : 0.0;

  // Calculate delta using the formula
  double delta = (excessCount * c1) + (disjointCount * c2) + (W * c3);

  return delta;
}

void Individual::mutateAddNeuron()
{
  // Ensure there are links to choose from
  if (m_genome.m_links.empty()) {
    return; // No connections to mutate
  }
  std::size_t index =
      m_config.m_rng.uniform<std::size_t>(0, m_genome.m_links.size() - 1);
  ConnectionGene &selectedConnection = m_genome.m_links[index];

  // Call addNode with the selected connection gene
  m_genome.addNode(selectedConnection);
}

void Individual::mutateAddLink()
{
  // Ensure there are neurons to choose from
  if (m_genome.m_neurons.size() < 2) {
    return; // Not enough neurons to connect
  }

  // Select two random neurons
  std::size_t fromIndex =
      m_config.m_rng.uniform<std::size_t>(0, m_genome.m_neurons.size() - 1);
  std::size_t toIndex =
      m_config.m_rng.uniform<std::size_t>(0, m_genome.m_neurons.size() - 1);

  // Ensure we don't connect a neuron to itself
  while (fromIndex == toIndex) {
    toIndex =
        m_config.m_rng.uniform<std::size_t>(0, m_genome.m_neurons.size() - 1);
  }

  // Create a new connection
  ConnectionGene newConnection(
      m_genome.m_neurons[fromIndex].m_neuron_id,
      m_genome.m_neurons[toIndex].m_neuron_id,
      0.0,  // weight can be initialized to 0.0 or some random value
      true, // enabled
      -1    // TODO: innovation ID to be determined
  );

  // Add the new connection to the genome
  m_genome.addConnection(std::move(newConnection));
}

void Individual::mutateRemoveNeuron()
{
  // Ensure there are neurons to choose from
  if (m_genome.m_neurons.size() <= m_config.m_num_inputs) {
    return; // Can't remove input neurons
  }

  // Select a random neuron that is not an input neuron
  std::size_t index =
      m_config.m_rng.uniform<std::size_t>(0, m_genome.m_neurons.size() - 1);

  // Remove the neuron from the genome
  NodeGene &selectedNeuron = m_genome.m_neurons[index];
  m_genome.removeNode(selectedNeuron);
}

void Individual::mutateRemoveLink()
{
  // Ensure there are links to choose from
  if (m_genome.m_links.empty()) {
    return; // No connections to remove
  }

  // Select a random connection gene
  std::size_t index =
      m_config.m_rng.uniform<std::size_t>(0, m_genome.m_neurons.size() - 1);

  // Remove the connection from the genome
  ConnectionGene &selectedLink = m_genome.m_links[index];
  m_genome.removeConnection(selectedLink);
}

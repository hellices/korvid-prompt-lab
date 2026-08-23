{
  "Name": "Korvid Prompt Lab Grounding Kubernetes Access",
  "Description": "Read Ollama discovery resources and open a pod port-forward in the ollama namespace.",
  "Actions": [],
  "NotActions": [],
  "DataActions": [
    "Microsoft.ContainerService/managedClusters/apps/deployments/read",
    "Microsoft.ContainerService/managedClusters/endpoints/read",
    "Microsoft.ContainerService/managedClusters/pods/read",
    "Microsoft.ContainerService/managedClusters/pods/write",
    "Microsoft.ContainerService/managedClusters/services/read"
  ],
  "NotDataActions": [],
  "AssignableScopes": ["__SUBSCRIPTION_SCOPE__"]
}

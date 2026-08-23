#!/usr/bin/env bash

validate_reflection_model() {
  local model="$1"
  [[ "$model" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._:/-]*$ ]]
}

reflection_provider() {
  local model="$1"
  printf '%s' "${model%%/*}" | tr '[:upper:]' '[:lower:]'
}

reflection_requires_credential() {
  case "$(reflection_provider "$1")" in
    ollama|ollama_chat) return 1 ;;
    *) return 0 ;;
  esac
}

reflection_credential_env_name() {
  case "$(reflection_provider "$1")" in
    openai) printf 'OPENAI_API_KEY' ;;
    anthropic) printf 'ANTHROPIC_API_KEY' ;;
    cohere) printf 'COHERE_API_KEY' ;;
    gemini|google) printf 'GEMINI_API_KEY' ;;
    ollama|ollama_chat) return 1 ;;
    *) printf 'OPENAI_API_KEY' ;;
  esac
}

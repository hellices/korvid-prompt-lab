FROM ghcr.io/astral-sh/uv:0.10.9 AS uv
FROM acrpensionguard.azurecr.io/runner-base:v1

USER root

RUN az aks install-cli \
      --kubectl-version v1.35.6 \
      --kubelogin-version v0.2.19 \
    && kubectl version --client \
    && kubelogin --version

COPY --from=uv /uv /uvx /usr/local/bin/

RUN uv --version \
    && az version \
    && git --version \
    && bash --version \
    && python3 --version \
    && jq --version

USER runner

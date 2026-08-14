# syntax=docker/dockerfile:1

FROM ros:humble-ros-base-jammy

ARG USERNAME=gj64
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble

# Install development and ROS tools while still running as root.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        sudo \
        build-essential \
        ccls \
        git \
        gdb \
        python3-colcon-common-extensions \
        python3-pip \
        python3-rosdep \
        vim \
        clang \
        clang-format \
        clang-tidy \
        ripgrep \
        ros-humble-foxglove-bridge \
        libpcl-dev \
        ros-humble-pcl-conversions \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root development user.
RUN groupadd --gid "${USER_GID}" "${USERNAME}" \
    && useradd \
        --uid "${USER_UID}" \
        --gid "${USER_GID}" \
        --create-home \
        --shell /bin/bash \
        "${USERNAME}" \
    && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" \
        > "/etc/sudoers.d/${USERNAME}" \
    && chmod 0440 "/etc/sudoers.d/${USERNAME}"

# Automatically source the base ROS installation in interactive terminals.
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" \
        >> "/home/${USERNAME}/.bashrc" \
    && chown "${USERNAME}:${USERNAME}" \
        "/home/${USERNAME}/.bashrc"

ENV DEBIAN_FRONTEND=

USER ${USERNAME}

WORKDIR /workspaces/Perception-Project

CMD ["bash"]

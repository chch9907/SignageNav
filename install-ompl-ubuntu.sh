#!/bin/bash

set -e

if [ `id -u` == 0 ]; then
    SUDO=
    export DEBIAN_FRONTEND=noninteractive
    apt-get -y install lsb-release
else
    SUDO="sudo -H"
fi

ubuntu_version=`lsb_release -rs | sed 's/\.//'`

install_common_dependencies()
{
    # install most dependencies via apt-get
    # ${SUDO} apt-get -y update
    # ${SUDO} apt-get -y upgrade
    # We explicitly set the C++ compiler to g++, the default GNU g++ compiler. This is
    # needed because we depend on system-installed libraries built with g++ and linked
    # against libstdc++. In case `c++` corresponds to `clang++`, code will not build, even
    # if we would pass the flag `-stdlib=libstdc++` to `clang++`.
    ${SUDO} apt-get -y install g++ cmake pkg-config libboost-serialization-dev libboost-filesystem-dev libboost-system-dev libboost-program-options-dev libboost-test-dev libeigen3-dev libode-dev wget libyaml-cpp-dev
    export CXX=g++
    export MAKEFLAGS="-j `nproc`"
}

install_python_binding_dependencies()
{
    ${SUDO} apt-get -y install python${PYTHONV}-dev python${PYTHONV}-pip
    # install additional python dependencies via pip
    ${SUDO} pip${PYTHONV} install -vU https://github.com/CastXML/pygccxml/archive/develop.zip pyplusplus
    # install castxml
    if [[ $ubuntu_version > 1910 ]]; then
        ${SUDO} apt-get -y install castxml
    else
        wget -q -O- https://data.kitware.com/api/v1/file/5e8b740d2660cbefba944189/download | tar zxf - -C ${HOME}
        export PATH=${HOME}/castxml/bin:${PATH}
    fi
    ${SUDO} apt-get -y install libboost-python-dev
    if [[ $ubuntu_version > 1710 ]]; then
        ${SUDO} apt-get -y install libboost-numpy-dev python${PYTHONV}-numpy
    fi
    if [[ $ubuntu_version > 1904 ]]; then
        ${SUDO} apt-get -y install pypy3
    fi
}

install_app_dependencies()
{
    ${SUDO} apt-get -y install python${PYTHONV}-pyqt5.qtopengl freeglut3-dev libassimp-dev python${PYTHONV}-opengl python${PYTHONV}-flask python${PYTHONV}-celery libccd-dev
    # install additional python dependencies via pip
    ${SUDO} pip${PYTHONV} install -vU PyOpenGL-accelerate
    # install fcl
    if ! pkg-config --atleast-version=0.5.0 fcl; then
        if [[ $ubuntu_version > 1604 ]]; then
            ${SUDO} apt-get -y install libfcl-dev
        else
            wget -O - https://github.com/flexible-collision-library/fcl/archive/0.6.1.tar.gz | tar zxf -
            cd fcl-0.6.1; cmake .; ${SUDO} -E make install; cd ..
        fi
    fi
}

install_ompl()
{
    if [ -z $APP ]; then
        OMPL="ompl"
    else
        OMPL="omplapp"
    fi
    if [ -z $GITHUB ]; then
        if [ -z $APP]; then
            wget -O - https://github.com/ompl/${OMPL}/archive/1.6.0.tar.gz | tar zxf -
            cd ${OMPL}-1.6.0
        else
            wget -O - https://github.com/ompl/${OMPL}/releases/download/1.6.0/${OMPL}-1.6.0-Source.tar.gz | tar zxf -
            cd $OMPL-1.6.0-Source
        fi
    else
        ${SUDO} apt-get -y install git
        git clone --recurse-submodules https://github.com/ompl/${OMPL}.git
        cd $OMPL
    fi
    mkdir -p build/Release
    cd build/Release
    cmake ../.. -DPYTHON_EXEC=/usr/bin/python${PYTHONV}
    if [ ! -z $PYTHON ]; then
        # Check if the total memory is less than 6GB.
        if [ `cat /proc/meminfo | head -1 | awk '{print $2}'` -lt 6291456 ]; then
            echo "Python binding generation is very memory intensive. At least 6GB of RAM is recommended."
            echo "Proceeding with binding generation using 1 core..."
            make -j 1 update_bindings
        else
            make update_bindings
        fi
    fi
    make
    ${SUDO} make install
}

for i in "$@"
do
case $i in
    -a|--app)
        APP=1
        PYTHON=1
        shift
        ;;
    -p|--python)
        PYTHON=1
        shift
        ;;
    -g|--github)
        GITHUB=1
        shift
        ;;
    *)
        # unknown option -> show help
        echo "Usage: `basename $0` [-p] [-a]"
        echo "  -p: enable Python bindings"
        echo "  -a: enable OMPL.app (implies '-p')"
        echo "  -g: install latest commit from main branch on GitHub"
    ;;
esac
done

# the default version of Python in 17.10 and above is version 3
if [[ $ubuntu_version > 1704 ]]; then
    PYTHONV=3
fi

install_common_dependencies
if [ ! -z $PYTHON ]; then
    install_python_binding_dependencies
fi
if [ ! -z $APP ]; then
    install_app_dependencies
fi
install_ompl

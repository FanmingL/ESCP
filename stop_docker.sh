#! /bin/bash
docker container stop $(docker container ps -aq --filter "ancestor=sanluosizhou/selfdl:latest")
docker container stop $(docker container ps -aq --filter "ancestor=sanluosizhou/selfdl:ml")
docker container prune -f
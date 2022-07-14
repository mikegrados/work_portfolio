FROM neo4j

RUN mkdir src
RUN mkdir src/data
RUN mkdir src/logs
RUN mkdir src/conf
RUN mkdir src/plugins

COPY ./data src/data
COPY ./logs src/logs
COPY ./conf src/conf
COPY ./plugins src/plugins

RUN neo4j-admin import --database=neo4j --nodes=src/data/idm-materias.csv --relationships=src/data/idm-relationships.csv --nodes=src/data/idm-semestres.csv --relationships=src/data/idm-mat-sem-rels.csv

ENV NEO4JLABS_PLUGINS=["apoc","graph-algorithms"]
ENV NEO4J_AUTH=none

CMD ["neo4j"]

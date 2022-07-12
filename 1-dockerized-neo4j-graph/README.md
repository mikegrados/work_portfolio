## README

The information about the courses was obtained in the following resources
- <a href="https://samp.itesm.mx/Materias/VistaPreliminarMateria?clave=F1010&lang=ES">SAMP</a>
- <a href="http://sitios.itesm.mx/va/planes_de_estudio/Catalogo_de_Programas_de_Profesional.pdf">Planes de estudio</a>

To build the image from the Dockerfile
> docker build . -t=**image_name**:dev

To run a container
> docker run -d -p 7474:7474 -p 7687:7687 **image_name**:dev

To see the graph
> MATCH(n) RETURN n
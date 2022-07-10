## README

The information about the courses was obtained in the following resources
- https://samp.itesm.mx/Materias/VistaPreliminarMateria?clave=F1010&lang=ES
- http://sitios.itesm.mx/va/planes_de_estudio/Catalogo_de_Programas_de_Profesional.pdf

To build the image from the Dockerfile, run the following in terminal
> docker build . -t=**image_name**:dev

To run a container, run the following in terminal
> docker run -d -p 7474:7474 -p 7687:7687 **image_name**:dev
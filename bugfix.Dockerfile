FROM --platform=linux/amd64 gcfntnu/scanpy:1.9.2

MAINTAINER Yuyao Song

RUN pip3 install harmonypy scvi-tools celltypist leidenalg argparse requests scimilarity scrublet

RUN pip3 install pydantic

ENTRYPOINT ["python"]
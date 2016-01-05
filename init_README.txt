init.sh é um arquivo script shell para inicializar o uso da GPU do theano.

Ele chama o python instalado no caminho:
/home/kenedos/anaconda/bin/python
e executa o arquivo python:
KerasExamples/cpu_gpu_test.py

Após isso, ele volta para a pasta home usando o cd ..
e entra na pasta do theano para remover arquivos temporários que bugam a execução da GPU.

É necessário executar init.sh com o comando
$ sudo bash init.sh

no terminal para que qualquer outro arquivo python do theano possa usar a GPU para execução.

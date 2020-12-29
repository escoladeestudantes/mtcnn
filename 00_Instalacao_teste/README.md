<div style="text-align:center"><a href="https://www.youtube.com/watch?v=45hakmPnTyo"><img src="https://i.imgur.com/61NnxY0.jpg"/></a></div>

<h4>MTCNN – Instalação e detecção facial com TensorFlow 2.0 e Keras 2.3.0</h4>

```
$ pip install mtcnn
```

<p>TensorFlow 2.0.0 não executa o código com a GPU, pois é mostrada a mensagem </p>

<ul><li>https://stackoverflow.com/questions/63542803/no-module-named-tensorflow-keras-layers-experimental-preprocessing</li></ul>

<p>Testando com uma versão mais antiga do Keras que é instalada junto com o mtcnn</p>


```
$ pip install keras==2.3.0
```
Primeiro teste funcionou com (https://stackoverflow.com/questions/57062456/function-call-stack-keras-scratch-graph-error)

If you use Tensorflow-GPU, then add:

```
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("physical_devices-------------", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
```

<p>Referências</p>
<ul>
<li>https://github.com/ipazc/mtcnn</li>
<li>https://stackoverflow.com/questions/63542803/no-module-named-tensorflow-keras-layers-experimental-preprocessing</li>
<li>https://stackoverflow.com/questions/57062456/function-call-stack-keras-scratch-graph-error</li>
</ul>


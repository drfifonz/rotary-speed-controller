<center>

Rotary speed controller
====

</center>

About
----
Students's project of automation model of DC motor with RL circuit controlled by rotational speed.</br>
Authors:</br>
- [Jakub Grabowski](https://github.com/jakgrab)
- [Jan Sibilski](https://github.com/JanSibilski)
- [Filip Patyk](https://github.com/drfifonz)

---
Mathematical abstract
----
<center>

![alt-text][model]

</center>

To be added.
---
How to run
----

Create conda enviroment using following command:
```bash
conda env create -f env.yml
```
Start local http server by runnig following command:
```bash
python src/server.py
```
Visualization is avalible at `localhost:8000/graph`

[model]: data/model.png "Model diagram"
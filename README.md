# Taller 1 — Dynamic Programming

Implementación de **Dynamic Programming** aplicada a un entorno de taxi inspirado en la ciudad de Milán.

El proyecto modela el problema como un **Markov Decision Process (MDP)** y aplica los algoritmos:

- Policy Evaluation
- Policy Iteration
- Value Iteration

## Entorno

El entorno está implementado con `Gymnasium` mediante la clase `MilanTaxiEnv`.

El estado se representa como:

~~~text
(row, col, pass_idx, dest_idx)
~~~

donde:

- `row`: fila actual del taxi.
- `col`: columna actual del taxi.
- `pass_idx`: ubicación del pasajero.
- `dest_idx`: destino del pasajero.

Configuración principal:

~~~text
Grid: 6 × 6
Estados: 720
Acciones: 6
p = 0.2
γ = 0.99
~~~

El número de estados se obtiene mediante:

~~~text
6 × 6 × 5 × 4 = 720
~~~

## Acciones

El agente dispone de seis acciones:

~~~text
0 → SOUTH
1 → NORTH
2 → EAST
3 → WEST
4 → PICKUP
5 → DROPOFF
~~~

## Recompensas

~~~text
Movimiento            → -1
Pickup incorrecto     → -10
Dropoff incorrecto    → -10
Entrega correcta      → +20
Alejarse del objetivo → -5
Acercarse al objetivo → +5
~~~



## Dinámica estocástica

El parámetro `p` introduce incertidumbre en las acciones de movimiento.

Con:

~~~text
p = 0.2
~~~

la acción seleccionada tiene una probabilidad de:

~~~text
1 - p = 0.8
~~~

Las otras tres acciones de movimiento tienen cada una:

~~~text
p / 3
~~~

## Markov Decision Process

El problema se representa como:

~~~text
MDP = (S, A, P, R, γ)
~~~

donde:

~~~text
S → Estados
A → Acciones
P → Probabilidades de transición
R → Recompensas
γ → Factor de descuento
~~~

La construcción del modelo se encuentra en:

~~~text
src/rl_project/models/mdp.py
~~~

## Matriz de transición P

La matriz `P` representa las posibles transiciones del entorno.

Para un estado `s` y una acción `a`:

~~~text
P(s' | s, a)
~~~

representa la probabilidad de pasar al estado `s'`.

Las transiciones se almacenan mediante:

~~~text
P[s][a]
~~~

Cada transición contiene:

~~~text
(probabilidad, siguiente_estado, recompensa, terminado)
~~~

## Matriz de recompensa R

La función de recompensa se representa mediante:

~~~text
R(s,a)
~~~

Para acciones estocásticas, la recompensa esperada puede expresarse como:

~~~text
R(s,a) = Σ P(s'|s,a) r(s,a,s')
~~~

`P` y `R` son utilizados posteriormente por los algoritmos de Dynamic Programming.

## Policy Evaluation

Policy Evaluation calcula el valor de los estados para una política determinada:

~~~text
Vπ(s)
~~~

Utiliza la ecuación de Bellman:

~~~text
Vπ(s) =
Σa π(a|s) Σs' P(s'|s,a)
[R(s,a) + γVπ(s')]
~~~

Durante el proceso se registra:

~~~text
||V(k+1) - V(k)||∞
~~~

para analizar la convergencia de los valores mediante diferentes `sweeps`.

## Policy Iteration

Policy Iteration alterna entre dos procesos:

~~~text
Policy Evaluation
        ↓
Policy Improvement
        ↓
Policy Evaluation
        ↓
Policy Improvement
        ↓
       ...
~~~

La mejora de política utiliza el valor esperado de cada acción:

~~~text
Q(s,a) =
R(s,a) +
γ Σs' P(s'|s,a)V(s')
~~~

y actualiza la política seleccionando las acciones con mayor valor.

## Value Iteration

Value Iteration actualiza directamente la función de valor:

~~~text
V(k+1)(s) =
max_a [
R(s,a) +
γ Σs' P(s'|s,a)V(k)(s')
]
~~~

Una vez obtenida la función de valor óptima, se obtiene la política:

~~~text
π*(s) = argmax_a Q(s,a)
~~~

El resultado de Value Iteration puede utilizarse posteriormente para controlar el taxi.

## Políticas

Las políticas se encuentran en:

~~~text
src/rl_project/policies.py
~~~

El proyecto incluye principalmente:

~~~text
RandomPolicy
ValueIterationPolicy
~~~

`RandomPolicy` selecciona acciones aleatoriamente.

`ValueIterationPolicy` utiliza la política calculada mediante Value Iteration para seleccionar las acciones del agente.

## Ejecución del agente

El agente sigue el siguiente flujo:

~~~text
env.reset()
     ↓
obtener estado
     ↓
policy.get_action(state)
     ↓
env.step(action)
     ↓
obtener nuevo estado y recompensa
     ↓
repetir
~~~

## Notebook

El desarrollo experimental principal se encuentra en:

~~~text
notebooks/milan_taxi.ipynb
~~~

El notebook contiene:

~~~text
1. Configuración del entorno
2. Construcción del MDP
3. Construcción de P
4. Construcción de R
5. Policy Evaluation
6. Policy Iteration
7. Value Iteration
8. Análisis de convergencia
9. Ejecución de políticas
~~~

## Instalación

### 1. Clonar el repositorio

~~~bash
git clone https://github.com/Cuervinho/Taller-1-Dynamic-Programming
cd Taller-1-Dynamic-Programming
~~~

### 2. Crear el entorno virtual

Linux/macOS:

~~~bash
python3 -m venv .venv
~~~

Windows:

~~~bash
python -m venv .venv
~~~

### 3. Activar el entorno

Linux/macOS:

~~~bash
source .venv/bin/activate
~~~

Windows PowerShell:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

### 4. Instalar dependencias

~~~bash
pip install -r requirements.txt
~~~

### 5. Instalar el paquete local

~~~bash
pip install -e .
~~~

### 6. Verificar la instalación

~~~bash
python -c 'import rl_project; print("Listo")'
~~~

Si la instalación es correcta:

~~~text
Listo
~~~

## Estructura del proyecto

~~~text
Taller-1-Dynamic-Programming/
│
│
├── notebooks/
│   └── milan_taxi.ipynb
│
├── src/
│   └── rl_project/
│       ├── envs/
│       │   └── milan_taxy.py
│       │
│       ├── imgs/
│       │   ├── custom_cliff/
│       │   └── custom_taxi/
│       │
│       ├── models/
│       │   └── mdp.py
│       │
│       ├── utils/
│       │   ├── taxi_utils.py
│       │   └── utils.py
│       │
│       ├── evaluation.py
│       ├── policies.py
│       └── train.py
│
├── pyproject.toml
├── README.md
└── .gitignore
~~~

## Archivos principales

| Archivo | Descripción |
|---|---|
| `milan_taxy.py` | Implementación del entorno Milan Taxi |
| `mdp.py` | Construcción del MDP, `P` y `R` |
| `policies.py` | Implementación de políticas |
| `taxi_utils.py` | Funciones de renderizado |
| `utils.py` | Funciones auxiliares |
| `evaluation.py` | Funciones de evaluación |
| `milan_taxi.ipynb` | Desarrollo principal del taller |

## Tecnologías

~~~text
Python
NumPy
Gymnasium
Matplotlib
Seaborn
Jupyter
PyTorch
Stable-Baselines3
Pillow
tqdm
~~~

## Autor

**Jheansel Hasler Beltrán Forero**
**Andres Felipe Cuervo Torres**

Proyecto académico de Reinforcement Learning — Dynamic Programming.
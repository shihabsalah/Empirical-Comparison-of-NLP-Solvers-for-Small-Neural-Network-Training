# Interactive Thesis Website Blueprint: Empirical NLP Solver Comparison

## Overview and Audience

This blueprint outlines a **single-page interactive website** for presenting an empirical comparison of nonlinear programming (NLP) solvers used in training a small neural network. The target audience is **mathematics professors** with limited machine learning background. Content will therefore be rigorous (grounded in math) but explained in **accessible, clear language** without assuming ML-specific knowledge. The page will be structured as a continuous scroll (a scrollytelling format), guiding the reader through conceptual explanations, mathematical derivations, and interactive visualizations.

Key design goals include:

* **Clarity for Non-ML Experts:** Explain concepts from first principles (e.g. what is a loss function, a gradient) assuming a strong math foundation but little exposure to neural networks. Avoid unexplained jargon.
* **Mathematical Rigor:** Provide the underlying mathematics (equations, derivations) behind each optimization algorithm. For example, show the update formulas and relate them to calculus concepts (gradient, Hessian).
* **Interactive Visualization:** Use engaging 3D graphics and animations to illustrate how each optimizer navigates an example loss landscape. The visuals should reinforce the text – e.g. showing the **step-by-step path** an algorithm takes down a surface, with arrows indicating step directions.
* **Emphasis on Comparison:** Throughout the page, highlight differences and **trade-offs** between optimizers – e.g. speed of convergence, path characteristics, how each chooses step size and direction. A final summary section will directly compare outcomes and performance side by side.

The user experience is a guided narrative: as the professor scrolls, new sections appear with text and visuals, building from background concepts to individual optimizer sections, and finally a comparative summary. The scroll interaction will also trigger animations (using scroll triggers or lazy loading) so that, for instance, when the “Gradient Descent” section comes into view, the corresponding 3D demo animates automatically. Users will also be able to interact with the 3D scenes (e.g. rotate or zoom the view) for deeper exploration, without requiring any complex inputs or coding.

## Technology Stack and Implementation

To create a modern, interactive, and AI-compatible experience, the site will be built with the following stack (avoiding unusual or niche dependencies):

* **React** (likely with a framework like Next.js for ease of development): React will manage the UI as a single-page application, enabling dynamic updates as the user scrolls. Each section of content can be a React component, possibly using a library like React Intersection Observer to handle scroll-triggered events (such as starting animations).
* **Three.js**: All 3D graphics and animations (e.g. loss surface plots, optimizer trajectories) will be rendered with Three.js. This allows us to display interactive WebGL content in the browser, such as a 3D loss landscape that the user can rotate. We will create one or more reusable Three.js components for the visualizations (for example, a component `<LossSurfaceScene>` that draws a surface and points/arrows for an optimizer’s path).
* **Tailwind CSS**: Tailwind will be used for styling the page with a clean, modern aesthetic. It will ensure consistency in typography, spacing, and layout without needing heavy custom CSS. We’ll use Tailwind’s utility classes for a **responsive design** that looks good on large screens (since professors might view on desktops) and is still usable on tablets.
* **Math Rendering**: To display equations clearly, we will integrate a math typesetting library (such as KaTeX or MathJax) with React. This allows us to include LaTeX equations in our Markdown/JSX. All important formulas (loss function definitions, update rules, etc.) will be rendered as high-quality math notation in-line with the text for readability.
* **Markdown/Content Integration**: For maintainability of heavy text sections, we may use MDX (Markdown + JSX) or a content library. This allows writing the explanatory text (including LaTeX for formulas) in Markdown, while embedding React components for interactive visuals. The site content (text and math) can thus be written in a Markdown file or Jupyter Notebook and imported, making it easier for collaborators (or AI agents) to edit content without touching low-level code. Jupyter notebooks can also be used offline to **precompute data** for visualizations (e.g. optimizer trajectories, example surface values) which are then loaded into the site as static JSON or embedded data. This ensures reproducibility without requiring heavy computation in the browser.
* **No Unusual Dependencies**: All libraries will be well-established (React, Three.js, Tailwind, KaTeX). We avoid any obscure plugins – this keeps the project easy for others to set up and ensures longevity. The only additional libraries might be small utility ones (e.g. `d3-scale` for color scales or lightweight UI libraries for tables), but generally we stick to the basics.

**Performance considerations:** The site will precompute and **pre-load any data/animations** needed. For example, the 3D coordinates of an optimizer’s path on a synthetic loss surface will be calculated in advance. Animations will then be basically playback of these precomputed paths, which is efficient. We’ll ensure Three.js scenes aren’t too heavy (limiting the mesh resolution of surfaces, number of points, etc., and pausing animations when offscreen). This makes the site responsive and avoids long loading times or requiring a GPU beyond a typical laptop’s capabilities.

Now we describe the content layout and components section by section:

## Background: Neural Network Optimization Problem

In this introductory section, we set up the **problem formulation** and context. We explain what it means to train a small neural network through the lens of an optimization problem, making it digestible for math professors.

**Content:**

* **Problem Definition:** We state that training a neural network is equivalent to solving a high-dimensional **nonlinear optimization** problem. For example, if \$w\$ represents all the parameters (weights) of the network, and \$L(w)\$ is the loss function measuring training error (e.g. a cross-entropy loss over the dataset), then training seeks to find \$w^\* = \arg\min\_w L(w)\$. This \$L(w)\$ is typically a **highly non-convex** function of \$w\$, meaning it has many local minima and saddle points rather than a single bowl-shaped minimum. We will illustrate what “non-convex” means visually in the next section.

  * We provide an example formulation for clarity: “Consider a neural network with parameters \$w \in \mathbb{R}^d\$. Given a training set \${(x\_i, y\_i)}\_{i=1}^N\$, the training loss can be written as
    $L(w) = \frac{1}{N}\sum_{i=1}^N \ell(f(w; x_i),\, y_i),$
    where \$f(w; x)\$ is the network’s output for input \$x\$, and \$\ell(\hat{y}, y)\$ is the loss for a prediction \$\hat{y}\$ (for instance, the cross-entropy loss for classification). We seek to minimize this \$L(w)\$.”
    This equation grounds the problem mathematically (and can be displayed with KaTeX). We then explain in words that \$L(w)\$ measures the discrepancy between the network’s predictions and the true labels, and training means adjusting \$w\$ to reduce this discrepancy.

* **Context of Small Neural Network:** We mention that in our scenario, the network is intentionally small (a modest Multi-Layer Perceptron on the MNIST digits dataset, as per the thesis). This is to make it feasible to run heavier optimization algorithms that would be too slow on larger networks. For instance, our network has on the order of \$10^5\$ parameters (one hidden layer with a few hundred neurons) – large enough to be interesting, but small enough to try sophisticated solvers. We won’t delve into the specifics of MNIST or the architecture deeply (since the focus is optimization, not the dataset), but we give just enough detail to set the stage.

* **First-order vs Second-order Methods:** We introduce the idea that there are many algorithms to solve the \$ \min L(w)\$ problem. The two broad classes are:

  * *First-order methods*: use only the gradient (first derivative) of \$L\$ to guide the search. These include **Gradient Descent (GD)** and variants like Stochastic Gradient Descent (SGD) and Adam. They make incremental updates proportional to the negative gradient.
  * *Second-order methods*: use both gradient and curvature (second derivative, Hessian) information to make more informed steps. These include **Quasi-Newton methods** like BFGS/L-BFGS and **Newton’s method** variants (including trust-region approaches). They can potentially converge in fewer iterations by accounting for the shape of \$L(w)\$.

  We explain that in modern deep learning, first-order methods (especially SGD with momentum and adaptive methods like Adam) are dominant because of their efficiency and scalability. However, for smaller problems we can experiment with more advanced NLP solvers (like L-BFGS or interior-point Newton methods) to see how they behave. This comparison can give insights into the **optimization landscape** and why certain algorithms perform better or worse.

* **Algorithms to be Compared:** We list the specific optimizers that the site will cover (which align with those in the thesis):

  1. **Gradient Descent (GD)** – basic first-order method.
  2. **Adam** – an adaptive first-order method (with momentum and per-coordinate learning rates).
  3. **Limited-memory BFGS (L-BFGS)** – a quasi-Newton second-order method.
  4. **Trust-Region Newton** – a Newton’s method that restricts step size (unconstrained trust-region approach).
  5. **Interior-Point (Trust-Constr)** – a variant of Newton’s method using an interior-point trust-region algorithm (suitable for constrained problems, applied here to an unconstrained problem).

  For completeness, we note that an ADMM-based solver was explored in the thesis work, but it yielded no improvement over GD and is omitted from our interactive presentation (to keep focus on the more illustrative methods).

**Visualization/Design:** This background section will primarily be text and formulas with minimal graphics, to avoid overloading the user at the start. However, we might include a small diagram or iconography for context:

* Perhaps a simple schematic of a neural network (to remind what the parameters \$w\$ live in) or a graph of a generic loss function. This could be a 2D curve of a loss vs parameter to illustrate “finding the minimum”. Since the next section covers 3D loss surfaces, we may save detailed visuals for that.
* The layout should be clean: a couple of paragraphs of text with formulas typeset nicely. Key terms like “gradient”, “Hessian”, “non-convex” could be bolded or linked to a tooltip definition for those who need a refresher.

By the end of this section, the reader should understand that we’re solving a challenging minimization problem and that different algorithms (to be introduced next) take different approaches to find the minimum.

## Loss Surfaces & Optimization Geometry

Before diving into each optimizer, this section provides an intuitive **visual explanation of loss surfaces and optimization geometry**. We want to build the reader’s mental model of the landscape these algorithms operate on – highlighting features like valleys, saddle points, ridges, and curvature – so that the later visualizations make sense.

**Content:**

* **High-Dimensional Loss Landscape:** We remind readers that \$L(w)\$ for a neural network is a function of potentially thousands of variables, which we cannot visualize directly. However, we can understand it via lower-dimensional analogies. We explain that one can think of \$L(w)\$ as a mountainous terrain or landscape in a high-dimensional space, where height = loss value and the horizontal axes represent directions in parameter space. **Local minima** are valleys or low basins (points where all small movements increase the loss), **saddle points** are flat passes or ridges (mixed curvatures: some directions go up, others go down), and **global minima** are the absolute lowest points. This sets up why optimization is hard: the landscape is not a simple bowl but can be very complex.

* **Curvature and Hessian:** We introduce the concept of curvature without heavy formula. For a math audience, we can mention the Hessian matrix (second derivative). A positive definite Hessian means a locally bowl-shaped surface (minima), indefinite Hessian means a saddle (since there are positive and negative curvature directions). We keep it conceptual: e.g. “In a valley, the loss might curve steeply upward in one direction but be flat or even curve downward in another – this is a saddle point scenario.” If helpful, we might give a quick 2D example: \$f(x,y) = x^2 - y^2\$ at (0,0) is a saddle: it’s a minimum in \$x\$ direction but a maximum in \$y\$ direction.

* **Visualization of Example Surfaces:** We present an **interactive 3D plot** of a **synthetic loss surface** to illustrate these concepts. Using Three.js, we will render a surface defined by some function \$z = f(x,y)\$ that has the features we want to show:

  * One possibility is a surface with a **narrow curved valley** leading to a minimum, and perhaps a saddle point on a ridge. For example, a modified **Rosenbrock function** (“banana function”) is ideal, as it famously has a long, narrow valley and is difficult for gradient descent. The global minimum sits at one end of the valley, but the curvature along the valley is much gentler than across the valley, illustrating ill-conditioning.

  * Alternatively, we could craft a custom function that has two or three “hills” and “valleys” to show multiple local minima and a saddle connecting them. The exact function can be decided in development (e.g. Rosenbrock: \$f(x,y)=(1-x)^2 + 100(y - x^2)^2\$, or something like \$f(x,y) = x^4 + x^2y^2 + y^2\$ which has a saddle at the origin).

  The **Three.js scene** will present this surface as a smooth shaded mesh. The user can **rotate** and **zoom** the 3D view (we’ll enable orbit controls) to examine the shape from different angles. Key features can be annotated: for instance, we might place a small flag or marker at a saddle point and at the global minimum. We can also color-code the surface by height (e.g. a heatmap or contour colors: red for high loss, blue for low loss) to make valleys vs hills distinct.

* **Explanatory text alongside visual:** As the user views this 3D plot, text will point out: “Notice the long valley on the left – this would correspond to a region where many weight configurations have similar medium loss. The small bowl at the end of the valley (marked by the star) is the global minimum – the very lowest loss. The peak in the middle is a local high region and the saddle (marked by the dot) is a point where if you move in one direction the loss goes up, but in a perpendicular direction it goes down.” This explanation ties directly to what an optimizer faces: e.g. a gradient-based method might get **stuck on a plateau or slow down in a flat valley**, and a second-order method might better handle the narrow curvature by “knowing” to take a longer step along the valley but a shorter step across it.

* **Dimension Reduction (mentioned conceptually):** We also foreshadow that later we will visualize actual optimizer *trajectories* on a loss surface. Since the real loss lives in high-D, we will use techniques like **Principal Component Analysis (PCA)** to project those trajectories into 2D or 3D. This is mentioned just to let the reader know that such projections are possible and meaningful (because optimizer paths often lie in a low-dimensional subspace of parameters). We won’t dive into PCA math here, just one sentence to set up the idea that “we can view the journey through parameter space in a simplified 3D projection.”

**Interactive Component Behavior:** This section’s 3D plot will be static in terms of not having an animation timeline (no optimizer moving yet), but it is interactive in that the user can rotate/inspect it. We will ensure that on loading, it shows a nice angle (probably an angled view to see the valley depth). There may also be UI toggles:

* Possibly a checkbox to toggle showing contour lines or a wireframe overlay on the surface.
* Perhaps a toggle to highlight curvature directions (for advanced users, e.g. show principal curvature axes at a point).
  These are optional and only if time permits – at minimum, a rotating 3D surface with color coding is great.

By the end of this section, the user should be comfortable with terms like “loss surface”, “valley”, “saddle” and be primed to see how different algorithms traverse such a landscape differently.

## Gradient Descent (GD)

Now the page transitions into sections focusing on each optimizer. The first is **Gradient Descent**, the simplest baseline method. We break this into an explanation of the algorithm and an interactive step-by-step visualization of it descending a loss surface.

### Concept and Math Explanation

We start by describing what Gradient Descent is and how it works, in intuitive terms and with an equation:

* **Basic Idea:** Gradient Descent is a **first-order iterative optimization** method that moves the parameters in the direction of the negative gradient (steepest decrease) of the loss. It’s like standing on a hill and always taking a step in the steepest downhill direction to reach the bottom.

* **Update Rule:** We present the mathematical update rule for GD:
  $w_{t+1} = w_t - \eta \,\nabla L(w_t)\,. $
  Here \$\nabla L(w\_t)\$ is the gradient of the loss at the current parameters, and \$\eta\$ is a small scalar called the **learning rate**. We’ll explain that \$\eta\$ controls the step size – too large and you might overshoot minima, too small and progress is slow.

  We note that in our experiments, we used a fixed learning rate (for example, \$\eta = 0.01\$) which gave stable if slow convergence. In practice, choosing \$\eta\$ is important and can be tricky: it often requires tuning or using schedules to decrease it over time, but in this simplified presentation we keep it constant for clarity.

* **Deterministic vs Stochastic GD:** Since our focus is on concept, we describe full-batch gradient descent (using the entire training set each step) as the idealized version. We might mention that Stochastic Gradient Descent (SGD) picks a random subset of data for each step which adds noise but is used in deep learning for efficiency. However, to keep things simple, our comparisons treat GD as if it were full-batch (deterministic) so we can fairly compare to methods like L-BFGS which also use full data. This means \$\nabla L(w\_t)\$ is the true gradient over all training data in our demos (no noise).

* **Convergence Characteristics:** We explain that gradient descent tends to require many small steps especially in ill-conditioned problems. If the loss function has a narrow valley (one direction steep, one direction flat), GD will **zig-zag** and progress slowly down the valley unless \$\eta\$ is extremely small. We can cite the Rosenbrock function example: *“For instance, the Rosenbrock function has a long, curved valley and gradient descent struggles to converge to the minimum because it keeps oscillating across the valley’s walls, making very slow progress along the valley floor.”* This sets up an expectation that GD is sensitive to the scale of the problem’s curvature.

* **No Second-Order Info:** We highlight that GD uses no Hessian/curvature information – it doesn’t “know” if a direction is shallow or steep except via the gradient magnitude. It also doesn’t adapt its step size during training in the basic version. The learning rate is fixed in advance, which is a limitation. We might mention that more advanced first-order variants (momentum, Nesterov, etc.) try to address some of these issues, but those are beyond scope as standalone entries (though Adam will incorporate momentum).

* **Use in our Comparison:** We note that we treat “Gradient Descent” as our baseline optimizer in the experiments. It’s essentially SGD with no momentum and a fixed learning rate on the full batch. We emphasize this is the simplest method to implement and understand, making it a good starting point.

### Interactive Visualization Walkthrough

After understanding GD conceptually, the site will show an **interactive 3D demo** of gradient descent on a sample loss surface. This will concretely illustrate how GD moves step by step.

**Setup:** We reuse the **same synthetic 3D loss surface** introduced earlier (for consistency, all optimizers will be visualized on this same surface). For example, suppose it’s a Rosenbrock-like valley surface. We mark an **initial point** on the surface representing the starting parameters \$w\_0\$. Typically, this will be a point with a fairly high loss (on a slope of a hill or somewhere above the valley).

We will have a graphical marker for the current point (e.g. a small sphere or a colored dot). The current point might be colored in a way that stands out (say bright yellow), and the target minimum could be marked with a star on the surface (so the user sees where it’s trying to go).

**Animation sequence:** When this section comes into view (or when the user clicks a “Play” button), an animation will play showing GD making a series of steps:

1. **Gradient Arrow:** At the starting point, we render an **arrow vector** on the surface showing the **negative gradient** direction. This arrow emanates from the point in the direction of steepest descent. The length of the arrow can be scaled for visualization (not necessarily exactly \$\eta |\nabla L|\$, or it might be scaled to match the actual step length for accuracy).
2. **Step Movement:** The point then moves along that arrow – this represents updating \$w \leftarrow w - \eta \nabla L(w)\$. We could animate the point sliding down along the arrow to its new position \$w\_1\$. This movement will be relatively small if \$\eta\$ is small.
3. The arrow at the old position can either disappear once the step is taken or remain as a trail. To help the user follow the path, we will **leave a trail**: perhaps a thin line or a series of small dots connecting the positions at each iteration, and/or keep past positions marked by faint spheres. This way the trajectory is visible.
4. At the new position \$w\_1\$, we again draw the arrow for \$-\nabla L(w\_1)\$ (the next step’s direction). Then move the point to \$w\_2\$, and so on.

We will simulate a number of iterations (for example, 10 iterations of GD) in this manner. The animation should be slow and clear enough that the user can see the direction changes. The user can pause/replay as needed. They can also rotate the 3D view during or after the animation to examine the geometry of the steps (the trail left behind is static in the scene).

**Visual highlights:** We explicitly demonstrate typical GD behavior:

* If the surface has a narrow valley, one might observe the point zig-zagging: the arrows might alternate directions, indicating GD corrects course each time instead of going straight down the valley. We can add a note in text like, “Notice how the path **zig-zags**: gradient descent always goes directly downhill locally, which can cause a back-and-forth across the valley if the minimum lies along a flat valley floor. It doesn’t ‘know’ to turn and go along the valley immediately.”
* We show that the step sizes are uniform in how we choose them (same \$\eta\$ each time), but the actual distance moved can shrink if gradients get smaller. In a well-behaved convex bowl, those steps might line up nicely towards the minimum; in a ravine, they might waste effort oscillating.

The on-screen annotation or caption might read: *“Gradient Descent takes small equal-sized steps in the direction of steepest descent at each point. In this example, the path (yellow trail) gradually descends, but you can see it doesn’t head straight to the goal – it follows the slope, which can lead to a **zig-zag path** in narrow valleys.”* We ensure to mention that if we made \$\eta\$ too large, the point would overshoot or oscillate wildly, and if too small, progress would be extremely slow – hence the importance of a proper learning rate.

From an implementation standpoint, the path for this demo will be precomputed (we can simulate GD on our chosen surface offline). We’ll store the sequence of \$(x,y)\$ positions and loss values. Three.js will be used to animate the marker along these points with a time delay between steps (or smoothly interpolate between them for a continuous move). The gradient arrows can be represented by 3D arrow objects (line with a cone tip). We might compute those gradients offline as well, or compute on the fly since it’s just our known analytic surface (computationally cheap). Either way, it’s deterministic and reproducible each time the animation runs.

By the end of this GD demo, the user will have a concrete picture of how plain gradient descent behaves on a loss landscape, setting the stage to contrast it with the more advanced methods next.

## Adam (Adaptive Moment Estimation)

Next, we cover **Adam**, a popular first-order optimizer that builds on gradient descent by adapting the learning rate for each parameter and incorporating momentum. We will explain Adam’s mechanics and then visualize how its trajectory might differ from plain GD.

### Concept and Math Explanation

* **What is Adam?** Adam is an **adaptive gradient method** that computes individual step sizes for each parameter based on estimates of first and second moments of the gradients. In simpler terms, Adam can be seen as a combination of **Momentum** and **RMSProp** (if the reader is familiar with those): it keeps track of a moving average of gradients and a moving average of squared gradients.

* **Motivation:** We explain in intuitive terms: Vanilla GD uses one global learning rate \$\eta\$ for all parameters and all directions. Adam tries to **adapt the step size for each direction** automatically. If a parameter’s gradient has been large on average, Adam will take smaller steps on that axis (to prevent oscillation), and if a gradient has been consistently small, Adam will take relatively larger steps (to ensure progress). This adaptation often leads to faster initial convergence and less sensitivity to the exact \$\eta\$ value.

* **Equations:** We provide Adam’s update equations (in a simplified form). Using notation:

  * \$m\_t\$ = first moment (mean of gradients), \$v\_t\$ = second moment (mean of squared gradients).
  * At each step \$t\$:
    $m_{t+1} = \beta_1\, m_t + (1-\beta_1)\,\nabla L(w_t),$
    $v_{t+1} = \beta_2\, v_t + (1-\beta_2)\,(\nabla L(w_t))^2,$
    where multiplications are element-wise for vectors. We mention typical values \$\beta\_1=0.9\$, \$\beta\_2=0.999\$, \$\epsilon = 10^{-8}\$ (to avoid division by zero).
    Adam then computes bias-corrected estimates \$\hat{m}*{t+1} = m*{t+1}/(1-\beta\_1^{t+1})\$ and \$\hat{v}*{t+1} = v*{t+1}/(1-\beta\_2^{t+1})\$ (which account for the initialization bias towards zero in early steps).
    Finally, the update is
    $w_{t+1} = w_t - \eta \frac{\hat{m}_{t+1}}{\sqrt{\hat{v}_{t+1}} + \epsilon}\,. $
    This formula can be shown in the text, possibly as “Algorithm Adam update (Equation)”, and we’ll cite the reference where it’s presented.

  We will explain each part in words: \$m\_t\$ is like a momentum term (exponentially decaying average of past gradients) which gives the direction a bit of memory (smoother than raw gradient). \$v\_t\$ is tracking the magnitude of gradients (like a variance) so we know if we’re consistently seeing big gradients in some direction. Dividing by \$\sqrt{v\_t}\$ means we **dampen the step** in directions with consistently high gradients (often steep directions) and relatively **amplify** steps in directions with tiny gradients (flat directions). The net effect is a more balanced, self-tuning update for each dimension of \$w\$.

* **Behavior and Benefits:** We mention that Adam often **converges faster initially** than plain GD because it automatically finds a good effective learning rate for each parameter. It can “zoom in” on relevant directions and not waste time oscillating on steep sides of valleys. It’s also more forgiving to the choice of global \$\eta\$ – one usually doesn’t have to tune as much as with GD. In our experiment, for fairness, we chose an \$\eta\$ for Adam that results in similar initial scale of updates as GD (e.g. if GD was 0.01, we might use 0.05 for Adam due to its internal averaging). But Adam will adjust internally either way.

* **Momentum effect:** We clarify that the momentum (\$m\_t\$) means Adam has a tendency to keep moving in the previous direction, which can accelerate progress along consistent descent directions (like going down a gentle slope repeatedly) and can smooth out noisy gradients. However, momentum also means Adam might overshoot minima if not careful, since it can build up speed.

* **Comparison with GD:** We explicitly say: unlike GD which uses the gradient at the current point in a straightforward way, Adam’s update is using a *transformed* gradient (scaled by past information). This often results in a **curved trajectory** in parameter space – Adam might veer in a slightly different direction than the immediate steepest descent, because its momentum carries it or its rescaling emphasizes certain directions. This can lead to interesting behaviors like overshooting and correcting, which we will see in the visualization.

### Interactive Visualization Walkthrough

We then demonstrate Adam on the same 3D loss surface, starting from the same initial point as GD, to contrast their paths.

**Setup:** The 3D surface scene is set up exactly as before. We have the same starting location for the parameter point. Now, we’ll animate the path taken by Adam’s algorithm. We will use the typical hyperparameters (β1=0.9, β2=0.999, ε=1e-8) and the chosen learning rate (e.g. 0.05 as per our experiment config). The path is again precomputed for a certain number of iterations (maybe 10-15 steps of Adam, which might effectively cover a good portion of the valley if it accelerates).

**Animation Differences:** The animation sequence is similar to GD:

* We draw arrows for the **Adam update direction** at each step. Note: The direction of Adam’s update is effectively \$\hat{m}\_{t}/(\sqrt{\hat{v}\_t}+\epsilon)\$, which may not be exactly the raw \$-\nabla L\$ direction. We can visualize an “effective gradient” arrow that Adam follows.
* The point moves along that arrow to the new position. We again leave a trail of the path.

However, we expect some noticeable differences in the trajectory:

* **Initial acceleration:** Adam often starts with small steps but then accelerates as the momentum builds. So in the first couple of steps, the arrows might be roughly aligned with what GD did, but soon the \$m\_t\$ momentum accumulates and Adam might take a larger step in the same general direction, possibly overshooting a bit beyond where pure GD would land. The visualization might show the step 3 or 4 being longer or in a slightly different direction than the immediate gradient because Adam is combining gradient information.

* **Curved path:** If the loss valley curves, Adam’s path may cut a bit more directly or even overshoot the minimum and come back. In the thesis analysis, they noted that Adam’s trajectory can form a **loop or spiral in 3D** – overshooting in one direction then correcting in another. We may see in our synthetic example that Adam doesn’t strictly zigzag like GD; instead, it might swing out and then converge. For instance, as Adam gains speed, it might go slightly past the valley minimum and then have to correct course backwards (a damped oscillation).

* **Smaller oscillations:** Thanks to adaptive steps, if there are any oscillations, they tend to damp out. We might show that after overshooting, Adam’s \$v\_t\$ would have increased (due to large gradient signal when it overshot), thus it automatically reduces the step size and the oscillation diminishes, bringing it to settle near the minimum.

We will highlight these in the accompanying text: *“Watch how Adam’s path (blue trail) **initially accelerates** and then does a slight **loop** near the optimum. The momentum carried it further in one direction, causing a mild overshoot beyond the minimum, after which it corrects course. This behavior contrasts with GD’s slow zig-zag – Adam makes faster initial progress, thanks to adaptive learning rates, but you can see a bit of overshooting (the looping path) as a side effect of its momentum. Ultimately, it converges more quickly to the valley.”*

Visually, we can distinguish Adam’s path by color (say blue vs GD’s yellow) and possibly marker shape. But since in the website we’ll show them in separate sections (not overlaid simultaneously here), color is just aesthetic. We ensure consistency: if GD was yellow, Adam could be blue, L-BFGS green, etc., and use those colors again in the final comparison overlay. The arrows for Adam’s update might be drawn in the same color as the path.

One subtlety: showing the actual computation of \$m\_t\$ and \$v\_t\$ in the animation might be too much, but we could illustrate momentum qualitatively. For example, we might depict the \$m\_t\$ vector by a dashed arrow or a ghost marker that shows the direction of accumulated momentum. This might be too technical, so likely we skip it and just show net effect.

The user can again rotate the scene to compare with how GD’s path looked. If they remember the GD path (or we allow toggling GD vs Adam path in the same view for advanced comparison), they’d see Adam took a different route. However, in this per-section approach, we will just focus on Adam here. We can provide a **button to “Compare to GD”** that, if clicked, might overlay GD’s trail for reference or pop up a side-by-side small view. This is an optional feature but would reinforce the difference. If we do it, we’d need to keep GD’s data loaded. It might be simpler to leave direct comparison to the final section.

**Key takeaways in text:**

* Adam’s ability to **adapt step sizes** means it doesn’t get stuck as badly as GD on plateaus – it sped through initially much faster.
* The presence of **momentum** means the path is smoother and more directed, but can overshoot, unlike GD which is very cautious (never overshoots because it’s so slow).
* In terms of convergence, we note that in our actual results, Adam reached a much lower loss than GD in the same number of iterations and achieved far higher accuracy on the task (GD stagnated). However, Adam did not reach quite as high accuracy as L-BFGS or TR methods eventually, which we’ll discuss later.

This section gives the reader an understanding of how an “adaptive first-order” method improves on plain GD, setting the stage to now introduce second-order methods.

## Limited-Memory BFGS (L-BFGS)

Now we move to the second-order family, starting with **L-BFGS**. L-BFGS is a quasi-Newton method that approximates full Newton’s method with limited memory. We will explain its principles and then show how it takes a more **direct path** on the loss surface.

### Concept and Math Explanation

* **What is L-BFGS?** We begin by explaining the name: Limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm. It’s a mouthful, so we clarify: it’s an algorithm that builds an approximate **Hessian inverse** on the fly to guide optimization. In other words, L-BFGS tries to estimate the curvature of \$L(w)\$ (how the gradient changes) and use that to take more informed steps than simple gradient descent.

* **Quasi-Newton Approach:** We recall Newton’s method: if we had the true Hessian \$H = \nabla^2 L(w)\$, one step of Newton’s method would be \$w\_{new} = w - H^{-1}\nabla L(w)\$, which ideally jumps to the critical point of the local quadratic approximation. True Newton steps can converge in very few iterations if the function is quadratic-like near optimum. However, computing \$H\$ and inverting it is extremely expensive for large \$d\$ (here \$d\sim 10^5\$). BFGS is an algorithm that *updates an estimate of the inverse Hessian* using gradient information observed between iterations. **L-BFGS** is a memory-efficient version that doesn’t store a full \$d\times d\$ matrix (which would be impossible for large \$d\$) – instead it stores a few vectors from recent iterations (the “limited memory” part, e.g. the last \$m=10\$ gradient differences).

* **How L-BFGS works (briefly):** We will not derive BFGS formula fully (as that’s quite involved), but we describe:

  * It starts typically with an initial guess of the inverse Hessian (often just identity or scaled identity).
  * At each iteration, it obtains the current gradient \$\nabla L(w\_t)\$. It also has stored info from previous steps (like \$s\_{t-1} = w\_t - w\_{t-1}\$ and \$y\_{t-1} = \nabla L(w\_t) - \nabla L(w\_{t-1})\$).
  * Using these, L-BFGS updates its estimate of \$H^{-1}\$ (or directly uses a two-loop recursion to compute \$H^{-1}\nabla L\$ without storing full matrix).
  * The direction \$p\_t\$ it chooses is this approximate Newton direction: \$p\_t \approx -H^{-1} \nabla L(w\_t)\$ (with \$H^{-1}\$ being the L-BFGS inverse Hessian approximation).

* **Line Search:** We mention that unlike GD where we preset \$\eta\$, L-BFGS typically uses a **line search** to find an appropriate step length along \$p\_t\$. After computing the direction \$p\_t\$, it will try a full step or bigger and smaller steps to see which decreases the loss sufficiently, and pick the best (satisfying Wolfe conditions or similar). This means L-BFGS steps are more or less automatically sized – often much larger and more assertive than the fixed small GD steps, but guaranteed (by line search) to not overshoot too badly. In practice, this allows L-BFGS to take **fewer, more impactful steps** than GD. We can say: “Each L-BFGS iteration typically makes significantly more progress than a single GD step, due to the curvature guidance and line search ensuring we take as large a step as is prudent.”

* **Memory/Cost:** We acknowledge that L-BFGS has more overhead per iteration: storing vectors and doing some linear algebra, plus multiple function evaluations during line search. However, because it often converges in far fewer iterations, it can be efficient for problems up to a certain size. For our small network (\~100k parameters), L-BFGS is feasible and indeed quite effective. (We might note from the results: L-BFGS reached the optimum in maybe around 8-10 iterations in our experiment, which is equivalent to a handful of epochs, and even had the lowest runtime among second-order methods – something we’ll show later.)

* **Intuition:** We give a simple intuition: “Imagine you are moving in a valley; GD only knows the slope at your current spot. L-BFGS, however, uses information from previous steps to **infer the curvature of the valley** – it figures out that in the past few moves, gradients changed in a way that suggests ‘there is a curved valley here’. So it effectively says: *‘turn slightly and take a bigger step along the valley’*. Thus L-BFGS tends to go more **directly towards the minimum** rather than hugging the sides of the valley.” This sets the expectation that L-BFGS path will be more direct and smooth.

* **Our usage:** We mention that in our implementation we used PyTorch’s `optim.LBFGS` with a history size (memory) of 10. L-BFGS was run with full-batch updates (deterministic) and we let it perform a certain max number of iterations per call (with line search). These details aren’t critical for the site, but we might mention that it’s an “advanced solver provided in libraries” to indicate it’s not custom-coded from scratch.

### Interactive Visualization Walkthrough

Now, we visualize L-BFGS on the same loss surface.

**Setup:** Same initial point as GD and Adam sections. We’ll animate the path that L-BFGS takes. Notably, L-BFGS might reach near the optimum in far fewer steps than GD/Adam, so our animation might only have maybe \~5 steps for L-BFGS (depending on how we simulated it). In the actual run, perhaps L-BFGS took \~8 outer iterations to converge for our problem; we can simulate \~5-8 steps for demonstration.

**Animation specifics:**

* Since L-BFGS uses line search, the concept of a fixed arrow of a certain length per step is different. We will still illustrate each iteration in two parts: first, show the **search direction** computed, then show the point moving along that direction to the chosen new point.

* We can draw a long arrow from the current point in the direction \$p\_t = -H^{-1}\nabla L\$ (approximate). This arrow might be drawn longer than GD’s because L-BFGS is willing to take a bigger step if possible. We might not know exactly how far along that arrow it will go (that’s what line search decides), but for visualization, we could mark a suggested extent (maybe the full arrow).

* Then, perhaps highlight a point on that arrow where it stops. We could even animate a small dot sliding along the arrow and stopping at the chosen step length. But to keep things simple, we might just jump the point to the new location and shorten the arrow to that length.

* Alternatively, a clearer way: do a small animation of *trial steps*: L-BFGS might try a full step and then maybe half step, etc. Simulating the actual line search is complex, so instead we assume it picks the ideal directly. We just show the final chosen step.

* As before, a trail is left. But likely L-BFGS’s trail will be short and fairly straight towards the minimum.

**What to observe:**

* L-BFGS path should look **efficient and straight**. For example, if GD zig-zagged and Adam curved, L-BFGS might cut straight through the valley. The trail might look like a direct line into the valley and then perhaps a slight curve near the end (if at all). According to the thesis, L-BFGS took a relatively direct path without oscillation and converged smoothly.
* We will note that **fewer steps** were needed. Possibly we’ll show fewer iterations in the animation and maybe mention “(Converged in 5 iterations for this example)” on screen.

The text annotation might say: *“L-BFGS rapidly finds the right **direction and step size**. In the visualization, note how **few steps** it needs: the green path goes nearly straight into the valley towards the minimum. This is because L-BFGS has accumulated curvature information; essentially, it **predicts the shape** of the loss surface and makes a **large, confident jump** each iteration. There’s no zig-zag and no overshooting – each step is guided by a line search to be as large as possible while still decreasing loss.”*

We can highlight that in practice, L-BFGS reached the target loss very quickly in our experiments, achieving one of the best final accuracies (97-98%) and doing so faster (in terms of iterations and even wall-clock time) than the other second-order method. The math professors might be interested in why it’s not used always: we can briefly mention that “for small problems, L-BFGS is excellent, but for very large neural nets it’s not practical due to memory and computation – which is why first-order methods dominate large-scale deep learning. Our scenario is just small enough to let it shine.”

From a dev perspective, the precomputed path for L-BFGS could be obtained by simulating or taking actual iterations from a run. It will be just a few points. We can differentiate this animation by perhaps using a distinct color (say **green** trail and arrows for L-BFGS).

Additionally, to emphasize what’s happening under the hood, we might include a small on-screen note or graph: e.g. “Estimated curvature” or how step length changed. But perhaps not needed – keep it visual and qualitative.

## Trust-Region Newton (Unconstrained)

The next solver is **Trust-Region Newton** method. This is a variant of Newton’s method that uses a trust region approach instead of line search. We’ll explain how trust-region methods work and then show its trajectory.

### Concept and Math Explanation

* **Newton’s method refresher:** We remind that classical Newton’s method would solve \$\nabla^2 L(w\_t), p = -\nabla L(w\_t)\$ for the step \$p\$ (i.e., take step \$p = -H^{-1}\nabla L\$). If \$L\$ were quadratic, Newton’s method would jump to the minimum in one step. However, if we are not near a minimum or if \$H\$ is not positive definite, Newton steps can be problematic (they might go uphill or wildly overshoot).

* **Trust-Region idea:** Instead of relying on a line search after the fact, trust-region methods **limit the step length *a priori*** by defining a region around the current point within which we “trust” the quadratic approximation of \$L\$. We explain: “Trust-region Newton chooses a step \$p\_t\$ by approximately solving the optimization
  $\min_p \quad \nabla L(w_t)^T p + \tfrac{1}{2} p^T H_t\, p \quad \text{s.t. } \|p\| \leq \Delta,$
  where \$H\_t\$ is the Hessian (or an approximation) at \$w\_t\$, and \$\Delta\$ is the trust-region radius. In words, it finds the best step according to the local quadratic model of the loss, but **constrains the step size** to be not too large (within a radius \$\Delta\$).”

  We then describe how the algorithm proceeds: it initially has some trust radius (could be small or moderate). If the step taken yields a good reduction in actual loss (meaning the quadratic model was trustworthy), it will **increase** the trust-region size (becoming more bold). If the step failed to reduce loss as expected (model was not accurate far out), it **shrinks** the trust radius. This way, the method dynamically adjusts how far it is willing to jump.

* **Conjugate Gradient Solver:** In practice, for large \$d\$, solving the trust-region subproblem exactly is expensive. The SciPy implementation (trust-ncg) uses a **conjugate-gradient iterative solver** to find the Newton step (or a truncated step if hitting the trust-region boundary). We might not dive deep into that, but we can say it uses efficient linear algebra to handle the Hessian indirectly.

* **Robustness:** The trust-region approach is **robust against overshooting**. It won’t take a step that is too large for the current model of the function. This avoids the scenario where pure Newton might diverge if you’re in a region where the quadratic model is a poor fit (e.g., far from the minimum or at a saddle where Hessian is not positive definite). Trust-region essentially **prevents unstable jumps** by capping step length. Only when it’s confident (the model predicted improvement well) does it allow bigger steps.

* **When trust-region Newton shines:** If the problem is well-behaved and you eventually get into the “basin” of a convex area around a minimum, trust-region Newton can converge *extremely fast* – often taking a couple of steps to basically finish optimization (the famous quadratic convergence of Newton’s method). In our experiment, we observed that the trust-region method (SciPy’s `trust-ncg`) often solved the problem in just \~3-7 iterations to high precision, once it got going. It basically “rockets” to the solution once it gets close. However, each iteration is costly (solving linear equations involving Hessian).

* **Compute cost trade-off:** We mention that computing the Hessian for our neural network and solving for steps is heavy, but we can afford it for this small network. The time per iteration is much more than GD/Adam, but it needs far fewer iterations. Indeed, in our results trust-region Newton took e.g. \~25 seconds vs GD’s \~17 seconds for 20 epochs, but achieved a far lower loss. So in moderate size problems, this is a viable trade.

### Interactive Visualization Walkthrough

Now the fun part: showing how trust-region Newton (let’s call it **TR** for short in text) moves on the surface.

**Setup:** Same initial point on the loss surface. We will animate perhaps a few iterations of the trust-region solver. We might not even need many because it will reach near the minimum quickly.

**What to illustrate:**

* The key difference is that trust-region might sometimes take a **big leap** if it trusts the model, or a **small cautious step** if not. We can visualize the trust region itself as a circle (sphere) around the current point. For instance, draw a transparent sphere (in 3D) or circle (projected) around the current point representing the boundary of step allowed. However, this might clutter the 3D view. An alternative is to use an arrow and indicate if it hit the boundary:

  * If the solver takes a step exactly of length = \$\Delta\$ (trust-region radius), it means it hit the boundary (the full Newton step was beyond trust radius and got clipped). If it takes a smaller step than \$\Delta\$, then either that was the optimal within or it didn’t need the full radius.
  * We could show \$\Delta\$ as a line or radius indicator.

Given time, we might avoid drawing the sphere and instead narrate it. But maybe a faint sphere could be instructive for the first step to say "this is how far it’s allowed to go".

* **First iteration:** Often, the first Newton step could be huge (if Hessian is well-conditioned). But trust region might restrict it. Let’s say our initial trust radius is moderate. If the full Newton step is larger, the algorithm will clamp to \$\Delta\$. So the point might not go all the way Newton would have wanted, but still a decent distance.

* We draw an arrow for the computed Newton direction (like we did for L-BFGS, but here from true Hessian if we had it). If the arrow is very long, we then show the point moving to a nearer point (the intersection of that direction with the trust-region boundary).

* We can then mention “the step was limited by trust region radius”. If the actual loss reduction was good, next iteration \$\Delta\$ will be increased. Possibly the second step then can be full Newton step if needed.

* **Subsequent iterations:** By the second or third iteration, if near the basin, trust-region Newton might take a full Newton step right to near the optimum. Perhaps at iteration 2 the trust radius is expanded and it leaps almost directly to the minimum (within tolerance).

So, the trajectory might look like:

* Step1: a moderate move (maybe not as small as GD, but possibly not huge if trust radius was cautious).
* Step2: a **big jump** (once trust region grows) straight into the valley close to min.
* Then it might stop (converged). Or a tiny refining step if needed.

We can animate two or three moves accordingly:

1. Draw arrow (Newton direction) at start, move point partially (within Δ).
2. Next, enlarge Δ (we could visually show sphere bigger if we want) and arrow again, then move a big distance.
3. Show final position near minimum.

**Trail and color:** Use a distinct color, say **purple** trail for TR Newton. The trail likely will have just a couple segments, mostly straight.

**Text commentary:** *“Trust-region Newton’s path is extremely efficient. In this example, it reaches the minimum in just two major steps. Initially, it computes a Newton step (using second-order info). The algorithm limits how far it goes on the first step (the trust region), so it doesn’t overshoot. After confirming the model was accurate, it **takes a very large step** straight into the optimum on the next iteration. The purple path shows how direct this was – essentially a straight line to the bottom of the valley. There’s no oscillation or wandering. This reflects Newton’s capability of rapid convergence once it’s in the right region (indeed, we see it *rocket* toward the solution).”*

We also note: “However, each of those steps is computationally heavy – computing the Hessian and solving for the step is costly. So while it took only \~3 iterations, it still used more time overall than, say, 20 iterations of GD in our case (we’ll compare timings soon). Trust-region Newton basically front-loads the work per iteration to reduce the number of iterations.”

If possible, we can mention how many Hessian evaluations or CG iterations happened, but that may be too deep. Simpler: “Under the hood, it was solving large linear systems to determine those steps.”

One thing to highlight is trust-region Newton’s **guarantee of no overshoot**: in contrast to plain Newton (not shown here), which might have jumped out of the valley and increased loss, the trust-region prevented that by taking a controlled step. We can mention that benefit.

This visualization likely impresses the speed of convergence (the point essentially jumps to near optimum). It should complement the earlier ones by showing the extreme opposite of GD’s slow crawl.

## Interior-Point (Trust-Constr) Newton

Finally, we cover the **Interior-Point Trust-Region** solver (labeled “trust-constr” in SciPy), which is essentially a Newton method designed for constrained problems, applied here without actual constraints. We explain how it differs and why its performance is unique.

### Concept and Math Explanation

* **What is Interior-Point method?** Interior-point methods are typically used to handle constraints by applying a barrier or penalty that keeps iterates within the feasible region. In an unconstrained scenario, the interior-point algorithm still operates, but it’s solving essentially the same Newton step with some additional mechanism (like perhaps maintaining positive definiteness or some barrier terms that are inactive since no constraints).

* **trust-constr specifics:** SciPy’s `trust-constr` algorithm internally uses a trust-region approach *and* an interior-point approach for constraints. Since we did not impose any constraints on the network parameters (no bound constraints, etc.), it effectively becomes a variant of Newton’s method that still goes through the motions of an interior-point algorithm. This means it likely forms a Karush-Kuhn-Tucker (KKT) system (with no actual constraint equations aside from trivial ones) and solves it with a trust-region on the augmented system. In simpler terms: *“Interior-Point (IP) in our case is essentially another trust-region Newton method, but with extra overhead meant for constraints (which we didn’t use). It’s like carrying safety equipment you don’t end up needing – it makes the process more cumbersome without giving a benefit for our unconstrained problem.”*

* **Differences from standard trust-region Newton:** Because it’s built for constraints, the interior-point method is very conservative to ensure feasibility. It might take many **tiny steps** (especially if it thinks it needs to satisfy some condition). The thesis notes that the interior-point solver took many tiny trust-region steps to ensure “feasibility” (though trivial) and thus was **extremely slow**. In fact, it converged in terms of reaching the optimum, but took a *“glacial pace”* to do so. We explain this in plain terms: *Interior-point methods introduce a barrier parameter that is gradually reduced – if there were constraints, it prevents violating them. Without constraints, perhaps it still keeps a very tight trust-region by default, or has additional stopping criteria that make it slow.* The key result is it **did eventually reach the solution** (just as well as the trust-region Newton did), but with far more iterations and computational effort.

* **Usage scenario:** We clarify that normally, you wouldn’t use an interior-point algorithm for an unconstrained problem – you’d use the simpler trust-region Newton. We included it just to compare. The interior-point method might be beneficial if we had constraints (like “weights must be positive” or something), but since we didn’t, it’s an overkill approach.

* **In our experiment:** We ran the interior-point solver effectively for one major iteration (since it’s a one-shot solve until tolerance). It took the longest time by far \~5-10x slower than others, and achieved nearly the same final loss as trust-region Newton (a bit higher loss, slightly lower accuracy perhaps, but close). We can mention that: *“Interior-point achieved \~95% accuracy, very close to the others’ \~97%, but it took 300 seconds vs 10-25 seconds for the other second-order methods. This inefficiency is the price of its generality.”*

### Interactive Visualization Walkthrough

For the final solver, we illustrate what its trajectory looks like. According to analysis, it moves straight toward the optimum (since it has the same basic Newton direction idea) but at a **much slower pace** (lots of tiny steps).

**Setup:** Same initial point. Now, interior-point might conceptually take many small sub-steps (maybe the algorithm internally iterates). If we strictly did one “iteration”, SciPy might return the final answer after lots of internal iterations, which might be difficult to animate directly. Instead, we can simulate a representative behavior:

* We know the direction it will head is similar to trust-region Newton (towards the optimum).
* But we can mimic it as if it took like 20 small steps along that direction, instead of 2 big steps.

So the visualization could show a **slow-motion march** to the minimum:

* The point moves in a straight line towards the minimum, but stops short, then again moves a bit, etc. Perhaps a series of short moves all in roughly the same direction.
* Essentially, it could look like a finely segmented version of the trust-region Newton’s straight line path. For dramatic effect, we might animate it in more, smaller increments than actually needed, to give the impression of slowness.

Alternatively, we might not animate every micro-step (could be tedious). Another approach:

* Show the interior-point’s path as a straight line (like trust-region), but animate the *speed* of the point’s movement to be very slow. For example, the point moves continuously along the line from start to end, but at a slow constant speed, whereas trust-region Newton would have covered that distance in one jump. This might metaphorically represent it taking many internal iterations.

We might even include a note “(the interior-point solver effectively took \~100+ micro-steps along this line)”.

**Visual cues:**

* Color the interior-point path in, say, **red** (if not used yet) or some distinct color. But note our table earlier used red for “Interior-Point” possibly. Actually no, we haven’t assigned red; GD-yellow, Adam-blue, L-BFGS-green, TR-purple, we can do interior in **orange** or **red**.
* We can place a marker as well, but since it ends at same final region, might overlap with TR’s final star. We ensure to separate animations, so it’s fine.

**Text annotation:** *“The interior-point Newton method heads straight toward the optimum (orange path), much like the trust-region method did, but you can see the progress is **very slow**. In this illustration, it’s as if the algorithm is inching forward in many tiny increments. This is because the method is very conservative internally, taking many small steps to ensure all conditions are satisfied. Ultimately, it reaches the same valley, but at a **glacial pace**. This solver carried the extra burden of enforcing constraints (even though we had none), which made it significantly slower without improving the result.”*

We might not show arrow vectors here because the direction hardly changes (it’s basically the same arrow pointing to the minimum the whole time). Instead, we just emphasize the slow movement.

To implement, one could simply animate the point moving along a straight line from start to end, but over a longer duration or in small increments. Or have it stop at intermediate points (dropping small breadcrumbs) to mimic iterations.

**Comparison note:** This will nicely set up the summary that interior-point is not worthwhile for unconstrained problems except demonstrating what if we had a constraints scenario. It also completes the set of algorithms to be compared.

---

After covering all individual solvers, we proceed to a section that **compares them directly** with summaries and combined visualizations.

## Trajectory Visualization & Optimizer Comparison

This section serves as the culmination: we bring together the optimizers to compare their performance and characteristics side by side. It will include an **interactive combined trajectory view** and a **summary table of metrics**, along with text discussion interpreting everything.

### Combined Trajectories in PCA Space

Having shown conceptual paths on a toy surface, we now show the **actual trajectories** these optimizers took when training the real neural network (as per our empirical results). To visualize their paths through the actual high-dimensional weight space, we use the PCA projection technique discussed earlier.

**Visualization component:** We present a 3D scatterplot of the optimizer trajectories in a common coordinate frame:

* The axes are the top 3 principal components of all the recorded weight snapshots (or we could do separate PCA per optimizer but it’s more informative to have one frame – however, the thesis did separate PCA per optimizer for clarity. For an interactive site, we could allow toggling between views).

* Each optimizer’s path is plotted as a series of points connected by lines in this 3D PCA space. We differentiate them by color and marker:

  * GD: perhaps black or gray, with a triangle at start and star at end (as in the thesis figures).
  * Adam: blue, start triangle, end star.
  * L-BFGS: green, same markers.
  * Trust-Region: purple.
  * Interior-Point: orange.

  Or use a consistent scheme from earlier sections.

* The points can be labelled by iteration (maybe small numbers next to them or a tooltip showing “epoch 0,1,...T”). In the thesis fig, each point was an epoch and colored from dark (start) to light (end). We could do a similar effect: gradient color along each path from start to finish.

* We also might plot a representation of the **loss surface contours** in that PCA plane for one optimizer, as was done in the thesis figure 2 for L-BFGS. However, doing that for an interactive site is complex (we’d need to sample loss on a grid in PCA space). It might be too heavy to include interactive contour surfaces for each. Perhaps omit that due to complexity. Instead, we can rely on the conceptual surfaces we already showed.

**Interaction:** The user can rotate this combined 3D plot to examine how each optimizer moved in relation to each other. We could also allow filtering: e.g. checkboxes to toggle each optimizer’s trajectory on/off for clarity, or a slider to highlight one at a time. But a simple approach: show all four/five trajectories together with different colors. If it’s cluttered, maybe show 4 (excluding interior or excluding GD since GD hardly moved). But probably include all.

We should highlight key observations (these come straight from the thesis analysis and will impress the differences):

* **Distance traveled:** GD’s trajectory is very short – it basically stayed near the initial point (the cluster of GD points is small). In contrast, all others moved far into a different region of parameter space where loss is lower.
* **Path shape:** L-BFGS and Trust-Region took fairly direct paths into that region (almost straight lines in projection). Adam’s path went out in one direction then looped around – it’s longer and curved. We might mention from the 3D view of Adam that it formed a loop (consistent with overshooting).
* **Oscillation:** Adam’s slight loop vs L-BFGS’s monotonic approach.
* **Interior-point:** Possibly its path is almost in the same direction as trust-region but just slower; since if it converged to same optimum, its track might overlap with trust-region’s track but just maybe a more incremental progression. If we only saved start and end, it’d just be a dot; but maybe we have intermediate if we considered “iterations”. The thesis might not have plotted interior point on fig1, possibly they excluded ADMM and maybe interior? Actually fig1 shows 4 subplots (GD, Adam, L-BFGS, Trust-Region). They left out interior and ADMM there.

  For completeness, we might stick to those four in combined plot. Or include interior as well if data available (maybe not worth clutter).
  We could mention interior in text though: that it basically eventually went to the same area as trust-region, but slower.

We can incorporate an **animation** in this 3D PCA plot: a small sphere moving along each trajectory path simultaneously or sequentially. For instance:

* Press “Play All” and you see five colored dots leaving the origin (start) at the same time and moving along their respective paths (like a race). Because each algorithm took different numbers of iterations, we might normalize time to epochs. e.g. GD moves one tiny step every interval, Adam moves faster, L-BFGS and TR jump almost to end quickly. This could visually dramatize speed differences.

  Specifically, at frame 1:

  * TR dot might already be near finish (since it converged in maybe 3 iterations out of 20 allocated).
  * L-BFGS dot also leaps far.
  * Adam dot moves moderately.
  * GD dot barely moves.

  As animation progresses:

  * TR dot might stop at end by frame 3 and then stay (it’s done).
  * L-BFGS might reach end by frame \~8.
  * Adam might approach end by frame \~20 (if we consider it ran full 20 epochs).
  * GD dot moves slowly and by frame 20 it’s still far from end.

This animated “race” would be very illustrative of convergence speed differences. We can definitely describe this: trust-region's point rockets ahead from the start, L-BFGS also quickly advances, Adam lags initially but picks up speed, GD plods along and basically gets nowhere significant. Adam overshoots a bit and comes back (we might not see overshoot in combined if just moving monotonically along PCA line, but maybe a slight backtrack can be shown if PCA captured oscillation).
Interior-point’s dot might move slowly too, akin to GD’s pace (the thesis said ADMM looked like GD, interior-point was slow as well). If we included interior point in race, it would move straight but extremely slowly, finishing last.

However, implementing multi-dot animation is more complex. As an alternative, we can have preset animations for each and allow user to select which to animate. But a combined race is compelling. It can be done by synchronizing on a timeline of "epochs". We’ll mention the possibility conceptually.

Given this is blueprint, we describe what the user learns:
\*“In the interactive 3D plot below, you can toggle each optimizer’s trajectory (or play an animation of them moving). It’s striking to see their differences:

* Gradient Descent (black) hardly left the start – its path is short and stays in one region, indicating it made minimal progress.
* Adam (blue) travels a longer, curved path. It initially goes in one direction (rapidly, due to adaptive steps) but then loops as it corrects course near the optimum.
* L-BFGS (green) and Trust-Region (purple) shoot almost straight to the far region where the minimum lies. Their paths are relatively direct. L-BFGS’s path in particular is a smooth trajectory into the valley, with no oscillation.
* If we animate their motion by epoch, we see Trust-Region and L-BFGS sprint ahead, reaching the valley within a few iterations, whereas Adam and especially GD lag behind. GD basically wanders near its start (no significant movement), confirming its slow convergence. Adam moves faster than GD but you see it overshoot (go past the target and circle back).
* The interior-point solver (orange, not shown in the figure for clarity) would trace a path similar to Trust-Region’s direction but at GD’s speed – essentially a slow, straight crawl to the solution.”\*

We might accompany this with an **example figure** or embedded image for clarity. For instance, a static combined plot from the thesis (Figure 1 was a 2D PCA for each separately, but maybe we have a 3D one from figure 3 for Adam or something). We actually have an uploaded image `pca_path_3d_L_BFGS.png` (which might be a single trajectory example). Perhaps we include it as an example of such a trajectory.

&#x20;*Example: 3D PCA trajectory of the L-BFGS optimizer’s training path (projected onto three principal components). The path starts at the green triangle (initial weights) and ends at the red star (final weights). We see L-BFGS took a relatively **direct route** through weight space, indicating an efficient convergence. In contrast, Adam’s trajectory (not shown here) would appear more curved or looped, and Gradient Descent’s path would be extremely short.*

*(In the actual website, an interactive 3D plot will allow the user to examine all optimizer trajectories in one view, as described.)*

The combination of this interactive visual and the earlier single-algorithm demos gives a comprehensive picture of *how* each optimizer navigates the landscape differently and correlates with performance.

### Performance Summary Table and Trade-off Discussion

Finally, we present a **comparison table** summarizing key performance metrics of each optimizer on the neural network training task. This table concisely shows the outcomes and costs, complementing the trajectory visualization with hard numbers.

We will include columns such as: Final Training Loss, Final Accuracy (on test set), Relative Loss vs GD, Relative Accuracy vs GD, Time taken, and Memory usage – similar to what was in the thesis. For example:

| **Optimizer**         | **Final Loss** | **Final Accuracy** | **Loss / GD** | **Accuracy / GD** | **Time (s)** | **Memory (MB)** |
| --------------------- | -------------: | -----------------: | ------------: | ----------------: | -----------: | --------------: |
| Gradient Descent (GD) |         2.2597 |              16.8% |          1.00 |              1.00 |         17.3 |             416 |
| Adam                  |         0.3898 |              88.5% |          0.17 |              5.26 |         17.4 |             416 |
| L-BFGS                |         0.2816 |              97.5% |          0.12 |              5.80 |         10.0 |             427 |
| Trust-Region Newton   |         0.2164 |              97.8% |          0.10 |              5.81 |         25.4 |             469 |
| Interior-Point        |         0.1806 |              95.1% |          0.08 |              5.65 |        299.5 |             438 |

*(Table: Performance of each solver at the end of training. “Loss / GD” and “Acc / GD” are ratios relative to Gradient Descent’s outcome. Time is wall-clock time for training, and Memory is peak usage.)*

We will explain the table to draw insights:

* GD ended with a very high loss and poor accuracy (essentially failing to train well in the allotted time). It’s our baseline = 1.0 in ratios.
* Adam achieved much lower loss (0.39) and high accuracy 88.5%, which is **5.3× the accuracy of GD**. It did so in roughly the same time as GD (both around 17 seconds) – showing its **efficiency per iteration** was higher (it made more progress in each epoch than GD). Memory usage was similar to GD.
* L-BFGS got even lower loss (0.28) and \~97.5% accuracy, essentially solving the problem almost completely. It interestingly took only \~10 seconds – faster than even GD/Adam. We note this is because it needed fewer iterations (though each iteration is costlier, apparently the GPU usage and our problem size made it quite fast). Memory a bit higher but not drastically.
* Trust-Region Newton also reached \~97.8% accuracy, slightly better loss (0.216), in about 25.4 seconds. So it was the second fastest method to reach top accuracy (wall-clock-wise, L-BFGS was fastest, then Adam/GD tied, then TR, with Interior last). Memory usage for TR was the highest (\~469 MB, due to Hessian factorization overhead).
* Interior-Point got 95.1% accuracy, a bit lower than TR/L-BFGS, with loss 0.1806 which is actually the lowest loss of all (overfit a bit maybe, or just slight differences). But its time was **\~300 seconds**, way off the chart – an order of magnitude slower than TR and two orders slower than L-BFGS, GD, Adam. Memory also a bit high. This confirms that the interior-point approach is impractical here.

We then discuss **trade-offs and conclusions**:

* **Second-order vs First-order:** Clearly the second-order methods (L-BFGS, TR) found a much better solution than plain GD in the same number of passes, confirming that curvature information can dramatically improve convergence quality. They achieved \~97% accuracy vs GD’s \~17%. Even Adam, an advanced first-order method, vastly outperformed GD (88.5% vs 16.8%) showing the importance of adaptive updates.
* **Adaptive (Adam) vs basic GD:** Adam’s fast initial progress stands out. If we had plotted loss vs time, we’d see Adam drops quickly early on (where GD was plodding). However, Adam didn’t ultimately reach the absolute lowest loss that L-BFGS and TR did. With more tuning or epochs, Adam might get closer, but by default it plateaued a bit shy of the optimum. This highlights that while Adam is great for speed, it might not always attain the very best final accuracy without careful tuning.
* **L-BFGS vs Trust-Region:** Both achieved excellent final results. L-BFGS was surprisingly efficient here – possibly due to GPU acceleration and fewer iterations. Trust-Region was also effective but had overhead. One interesting observation: L-BFGS had the advantage of using a well-optimized library and fewer iterations, so it was fastest. Trust-Region might scale better if higher precision is needed or constraints are present, but here L-BFGS was the winner in speed.
* **Interior-Point:** We note that while it did fine in final accuracy, it’s clearly not worth it for unconstrained problems due to massive time cost. It essentially demonstrates that more complex algorithms aren’t always better – one should match the solver to the problem. If we had constraints, interior-point might be necessary, but without them, it’s doing unnecessary work.

We wrap up the comparison with a few **takeaway bullet points** (for easy scanning by the professor):

* *Gradient Descent:* Easiest to implement but **converges slowly** and may fail to find a good minimum in limited time.
* *Adam:* **Much faster initial progress** than GD, robust to hyperparameters, but may **not reach the absolute best solution** without tuning. Great general-purpose choice for moderate accuracy quickly.
* *L-BFGS:* **Very fast convergence** in terms of iterations (took direct paths), and here even time-wise it was excellent. Achieved highest accuracy, but memory and computation per step are heavier; feasible for smaller models.
* *Trust-Region Newton:* **Few iterations to converge** and finds excellent solutions. More stable than plain Newton thanks to trust radius. But each iteration is expensive and requires Hessian; not scalable to big models, but a powerful method when applicable.
* *Interior-Point Newton:* **Not efficient for unconstrained problems** – included for academic curiosity. It converged correctly but with far more work. Shows the cost of generality in solver design (only consider it if constraints need to be strictly handled).

Finally, we would conclude with a short note reinforcing the educational value:
The interactive journey through these optimizers illustrated not just which is “fastest” or “best,” but *why* they behave differently – by visualizing their decision-making on a loss surface, we gained intuition. This site thus serves as a pedagogical tool bridging rigorous mathematical concepts (derivatives, Hessians, line searches, trust regions) with practical outcomes in neural network training. Math professors can appreciate how the theory of optimization manifests in a concrete ML task, and hopefully apply this understanding to other problems or teaching.

This blueprint has specified all content, layout, and interactive features in detail. With this plan, developers or AI agents can proceed to implement the website, confident that it delivers a comprehensive and engaging experience. All mathematical content and visualizations have been outlined, and the modern tech stack will ensure the site is responsive and visually appealing. The result will be an interactive thesis presentation that makes advanced optimization concepts clear and accessible, even to those new to machine learning, by leveraging their mathematical intuition.

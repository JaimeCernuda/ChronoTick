\section{Design}

ChronoTick is a software-defined clock system designed to maintain a high-fidelity view of physical time by proactively predicting and correcting clock drift in commodity systems. 

\subsection{Architecture}
\begin{figure}[h]
    \centering
    \includegraphics[width=\linewidth]{sections/Figures/Arch.png}
    \caption{ChronoTick system architecture. Continuous correction feeds time into applications via a presentation layer, while a forecasting engine predicts drift from collected system metrics and retrospective correction integrates external references.}
    \label{fig:arch}
\end{figure}

The architecture of ChronoTick, shown in Figure~\ref{fig:arch}, is composed of five core components that interact in a streaming, feedback-driven loop: the \textit{presentation layer}, \textit{data collection layer}, \textit{forecasting engine}, \textit{continuous correction engine}, and the \textit{retrospective correction module}.

At the heart of the system is the \textbf{Correction Engine}, which continuously synthesizes a corrected view of time from forecasted drift and offset values. This corrected time is exposed to the rest of the system through the \textbf{Presentation Layer}, which may take the form of a shared memory segment or embedded runtime access. Applications use this interface to access the clock view.

The \textbf{Data Collection Layer} operates asynchronously, harvesting time and system telemetry from the local node. These observations populate the \textbf{Time Dataset}, which is used by the \textbf{Forecasting Engine}. This engine leverages both a CPU-based, low-latency, low resources, short-term model and an optional GPU-based, resource demanding, long-term model to predict future drift patterns based on the dataset’s evolving state.

Finally, the \textbf{Retrospective Correction Module} is triggered upon receipt of a high-precision external synchronization point (such as NTP or PTP). It applies bias correction to recent history, updating the dataset and retraining the forecasting models if necessary. This allows ChronoTick to re-anchor its trajectory without introducing discontinuities, ensuring that future predictions remain well-calibrated.

This architectural flow enables ChronoTick to operate autonomously and continuously, offering a time service that is both responsive and robust—even in the presence of drift, transient conditions, or low-frequency synchronization events.


\subsection{Presentation Layer}

ChronoTick supports two modes of use: as a system-wide daemon or as an embedded library. In daemon mode, it runs as a privileged background process that exposes the adjusted time to other applications via a shared memory segment. This memory region contains the current corrected timestamp, computed by ChronoTick based on forecasted clock behavior. To ensure thread-safe and efficient access to this shared resource, the system employs a sequence lock (seqlock) synchronization mechanism. Seqlocks allow multiple readers to access the memory without blocking, as long as the data remains stable during the read. Writers increment a sequence counter before and after updating the memory region; readers verify that the counter is even and unchanged across the read window, retrying if a concurrent write is detected.

In embedded mode, ChronoTick can be linked directly into an application as a library. In this configuration, a module-level singleton is initialized during application startup and maintains the corrected time in-process. This approach is particularly useful for containers or environments where shared memory is inaccessible or undesirable.

\subsection{Collection Layer}

To drive accurate predictions, ChronoTick collects a variety of timing and system metrics at a regular interval aligned with the model's expected input frequency. A synchronization signal triggers those measurements. The collection process runs in a dedicated thread with affinity set to one CPU core, ensuring minimal interference from scheduling noise. Each sampling event can gather data from the following sources: the system clock, a high-resolution monotonic timer, the local clock synchronization daemon (NTP, PTP, etc), and a set of environmental sensors (e.g., temperature, voltage, frequency).

When a synchronization daemon is present (Chrony was used in our experiments), ChronoTick extracts detailed metadata that characterize the current behavior and reliability of the clock against the known reference. These metrics include the estimated offset between the system clock and the reference clock, the timestamp of the last synchronization, the frequency drift, the residual frequency after any corrections externally implemented, clock skew, root delay (indicating minimum network delay to the reference), root dispersion (representing accumulated clock error).

\subsection{Prediction Layer}

ChronoTick supports a modular prediction engine that incorporates multiple forecasting strategies, allowing it to balance performance and accuracy. 

A short-term model executes low-latency, high-frequency, CPU forecasting. This model is trained to predict short-term drift using historical timing and exogenous system metrics with the goal of capturing rapid fluctuations or anomalies in the system. Running on a second interval and producing few few-second forecasting. By its nature, this model requires a shorter memory context, the ability to operate with exogenous variables, and to be lightweight.

Complementing the short-term model, an optional long-term model is used. This model is designed, but not required, to operate under GPU acceleration, with infrequent, but more expensive inferances with a longer view when compared to the short-term model. 
Its goal is to provide a smooth and stable long-term view of the clock behaviour while missing the high-frequency detail that the short-term mode supports. The long-term model requires the ability to operate with longer context windows and generate longer views into the present. Ideally, the output should include a quantification of its uncertainty, a feature not demanded of the short-term model, where each forecast includes a central estimate and upper and lower quantiles (typically 10\% and 90\%), allowing ChronoTick to assess the reliability of short-term predictions.

% ChronoTick continuously compares TTM predictions to the confidence bounds provided by TimeFM. When TTM forecasts fall within the expected range, the system continues to trust them for high-speed correction. If TTM deviates outside the range, it triggers an early re-evaluation from TimeFM and may downweight or suppress the TTM forecast until revalidated. This mechanism ensures robustness while preserving responsiveness.

Building on this, ChronoTick leverages the uncertainty estimates provided by the long-term model to probabilistically combine both forecasting sources. When each forecast includes a central estimate along with quantiles---typically the 10\% and 90\% percentiles---ChronoTick interprets the spread between these quantiles as a measure of predictive uncertainty. Assuming approximate normality, the standard deviation can be estimated as:
\[
\sigma \approx \frac{Q_{90} - Q_{10}}{2.56},
\]
where $Q_{90}$ and $Q_{10}$ denote the 90th and 10th percentile forecasts, respectively. The combined forecast $\hat{y}$ at time $t$ is computed using inverse-variance weighting:
\[
\hat{y}(t) = \sum_i w_i(t) \cdot \hat{y}_i(t), \quad \text{with} \quad w_i(t) = \frac{1/\sigma_i^2}{\sum_j 1/\sigma_j^2}.
\]
This adaptive fusion allows ChronoTick to prioritize the more confident model at each timestep. When the short-term model exhibits low uncertainty, its high-frequency responsiveness dominates; when its forecasts become unreliable, the smoother and more stable long-term model anchors the prediction. This strategy ensures both temporal accuracy and long-horizon coherence in the system's behavior.


\subsection{Continuous Clock Correction}
ChronoTick synthesizes a corrected physical time using an offset-plus-drift model. Given a known offset  at time , and a later offset  at , the estimated drift rate  is computed as:
$$
\text{offset}(t) = \text{offset}(t_1) + \text{rate} \cdot (t - t_1)
$$

Using this, ChronoTick estimates the current offset  at time  via linear extrapolation:
The corrected system time  is then defined as:
$$
\text{corrected\_time}(t) = t + \text{offset}(t)
$$

This correction is recalculated at each sampling interval, using updated forecasts of  and  from the prediction engine. The adjusted time is published to the shared memory segment or returned by library calls, depending on the deployment mode. Corrections are applied incrementally to preserve monotonicity and avoid time regressions.

\subsection{Retrospective Correction Layer}

ChronoTick refines its recent predictions when a high-confidence external synchronization event occurs. Rather than applying a step-change to the current offset, it performs a retrospective smoothing over recent predictions. The goal of the algorithm, seen in \ref{algo:smoothing}, is to distribute the correction gradually across the most recent interval, under the assumption that prediction error accumulated over time.

Suppose that a synchronization event happened at time \( t \), (the previous sone having happened at \( t-1 \)). A discrepancy is observed between the model predicted offset \( \hat{o}_t \) and the true offset \( o_t \). Instead of applying the full correction only at \( t \), ChronoTick distributes this correction over the predictions within the interval \([t-1, t)\).

Each earlier timestamp \( t' \in [t-1, t) \) receives a fraction of the total correction depending on its temporal proximity to the synchronization time. Predictions made closer to \( t-1 \) are assumed to be more accurate and thus are corrected less, while predictions made closer to \( t \) are adjusted more, as they likely reflect the accumulated drift. To determine the strength of each correction, a weight \( \alpha_{t'} \in [0, 1] \) is assigned, increasing linearly over time. For example:
\begin{itemize}
  \item \( \hat{o}_{t-1}'': \alpha = 0 \) (no correction)
  \item \( \hat{o}_{t - 0.5}'': \alpha = 0.5 \)
  \item \( \hat{o}_{t - \epsilon}'': \alpha \approx 1 \)
\end{itemize}

For each timestamp \( t' \), the corrected prediction is computed by adding a proportion \( \alpha_{t'} \) of the total correction \( \delta_t = o_t - \hat{o}_t \) to the original offset \( \hat{o}_{t'} \). This method ensures a smooth transition and prevents abrupt discontinuities in the adjusted time series, preserving the continuity and trustworthiness of time estimates. It also reflects a realistic assumption that synchronization error builds gradually rather than instantaneously.

\begin{algorithm}[H]
\SetAlgoLined
\KwIn{True offset $o_t$ at time $t$, predicted offset $\hat{o}_t$,\\
\hspace{1.5em} Prior timestamps $\{t_i\}_{i=0}^{n}$ in $[t-1, t)$,\\
\hspace{1.5em} Predicted offsets $\{\hat{o}_{t_i}\}_{i=0}^n$}
\KwOut{Corrected offsets $\{\hat{o}_{t_i}''\}_{i=0}^{n}$}
$\delta \leftarrow o_t - \hat{o}_t$\;
\For{$i \leftarrow 0$ \KwTo $n$}{
    $\alpha \leftarrow t_i - t + 1$\;
    $\hat{o}_{t_i}'' \leftarrow \hat{o}_{t_i} + \alpha \cdot \delta$\;
}
\label{algo:smoothing}
\caption{Smoothing with Retrospective Bias Correction (SRBC)}
\end{algorithm}

After applying this correction, ChronoTick can optionally retrain or recalibrate its prediction models based on the newly adjusted history, improving forecast accuracy moving forward.

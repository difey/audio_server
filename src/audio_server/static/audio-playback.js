// Audio playback for transcription results.
// Intercepts "final" WebSocket messages containing audio (base64 PCM16),
// stores the audio per segment, and adds ▶ play buttons to result items.

;(function () {
  'use strict'

  // ── Audio storage & playback ──────────────────────────────────────
  const audioMap = new Map()
  let audioCtx = null

  function getAudioCtx() {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)()
    }
    if (audioCtx.state === 'suspended') audioCtx.resume()
    return audioCtx
  }

  function base64ToInt16(b64) {
    const raw = atob(b64)
    const buf = new ArrayBuffer(raw.length)
    const view = new Uint8Array(buf)
    for (let i = 0; i < raw.length; i++) view[i] = raw.charCodeAt(i)
    return new Int16Array(buf)
  }

  function playPCM16(int16) {
    const ctx = getAudioCtx()
    const f32 = new Float32Array(int16.length)
    for (let i = 0; i < int16.length; i++) f32[i] = int16[i] / 32768
    const ab = ctx.createBuffer(1, f32.length, 16000)
    ab.getChannelData(0).set(f32)
    const src = ctx.createBufferSource()
    src.buffer = ab
    src.connect(ctx.destination)
    src.start()
  }

  // ── DOM helpers ───────────────────────────────────────────────────
  function addPlayButton(el, startTime) {
    if (el.querySelector('.play-btn')) return
    const btn = document.createElement('button')
    btn.className = 'play-btn'
    btn.title = '播放此段音频'
    btn.textContent = '▶'
    Object.assign(btn.style, {
      background: 'none',
      border: '1px solid #4ade80',
      color: '#4ade80',
      borderRadius: '4px',
      cursor: 'pointer',
      fontSize: '14px',
      padding: '2px 8px',
      marginLeft: '8px',
      flexShrink: '0',
      transition: '.15s',
    })
    btn.onmouseenter = () => { btn.style.background = '#4ade8033' }
    btn.onmouseleave = () => { btn.style.background = 'none' }
    btn.onclick = (e) => {
      e.stopPropagation()
      const data = audioMap.get(startTime)
      if (data) {
        btn.textContent = '◼'
        playPCM16(data)
        setTimeout(() => { btn.textContent = '▶' }, 150)
      }
    }
    const container = el.querySelector('.result-main') || el.firstElementChild
    if (container) container.appendChild(btn)
  }

  function scanForButtons() {
    document.querySelectorAll('.result-item.final').forEach((el) => {
      const start = el.getAttribute('data-start')
      if (start && audioMap.has(parseFloat(start))) {
        addPlayButton(el, parseFloat(start))
      }
    })
  }

  // ── WebSocket construction interception ──────────────────────────
  // Use Proxy to avoid `new` / prototype issues with native WebSocket
  const OrigWS = window.WebSocket

  window.WebSocket = new Proxy(OrigWS, {
    construct(target, args) {
      const ws = new target(...args)
      ws.binaryType = 'arraybuffer'

      // Intercept onmessage setter to capture Vue's handler
      let vueHandler = null
      Object.defineProperty(ws, 'onmessage', {
        set(fn) { vueHandler = fn },
        get() { return vueHandler },
      })

      // Intercept messages at the event-listener level
      ws.addEventListener('message', function (evt) {
        if (typeof evt.data !== 'string') return // skip binary

        try {
          const msg = JSON.parse(evt.data)

          // ── final with audio: store, strip, forward to Vue ──
          if (msg.type === 'final' && msg.audio) {
            audioMap.set(msg.start, base64ToInt16(msg.audio))
            delete msg.audio
            if (vueHandler) vueHandler({ data: JSON.stringify(msg) })
            evt.stopImmediatePropagation()

            setTimeout(() => {
              const items = document.querySelectorAll('.result-item.final')
              const last = items[items.length - 1]
              if (last) {
                last.setAttribute('data-start', msg.start)
                addPlayButton(last, msg.start)
              }
            }, 50)
            return
          }
        } catch { /* ignore parse errors */ }

        // Non-audio messages: forward and prevent double-fire via onmessage
        if (vueHandler) vueHandler(evt)
        evt.stopImmediatePropagation()
      })

      return ws
    },
  })

  // Re-scan periodically for dynamically rendered results
  setInterval(scanForButtons, 500)
})()

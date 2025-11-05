import React, {useState, useRef} from 'react'
import axios from 'axios'

const API_BASE = "http://127.0.0.1:8000"

function UploadPanel({title, endpoint}){
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const inputRef = useRef()

  const onFile = (f)=>{
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setResult(null)
  }

  const onDrop = (e)=>{
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    onFile(f)
    uploadFile(f)
  }

  const uploadFile = async (f)=>{
    if(!f) return
    setLoading(true)
    const form = new FormData()
    form.append('file', f, f.name)
    try{
      const res = await axios.post(API_BASE + endpoint, form, {headers: {'Content-Type':'multipart/form-data'}})
      setResult(res.data)
    }catch(err){
      alert('Upload failed: ' + (err.response?.data?.detail || err.message))
    }finally{
      setLoading(false)
    }
  }

  return (
    <div className="panel">
      <h2>{title}</h2>
      <div className="uploader" onDragOver={(e)=>e.preventDefault()} onDrop={onDrop}>
        <p>Drag & drop an image here or click to select</p>
        <input ref={inputRef} type="file" accept="image/*" onChange={(e)=>{onFile(e.target.files[0])}} />
        <button onClick={()=>{ if(inputRef.current) inputRef.current.click() }}>Choose file</button>
      </div>

      {file && <div className="preview"><img src={preview} alt="preview" /></div>}
      <div className="controls">
        <button disabled={!file || loading} onClick={()=>uploadFile(file)}>Run Detection</button>
        <button onClick={()=>{ setFile(null); setPreview(null); setResult(null) }}>Clear</button>
      </div>

      {loading && <p>Processing on server...</p>}
      {result && result.success && (
        <div className="result">
          <h3>Annotated</h3>
          <img src={API_BASE + result.annotated_image_url} alt="annotated" />
          <h4>Detections</h4>
          <ul>
            {result.predictions.map((p,i)=>(<li key={i}>{p.label} — {p.confidence.toFixed(3)} — [{p.xyxy.map(v=>v.toFixed(1)).join(', ')}]</li>))}
          </ul>
        </div>
      )}
    </div>
  )
}

export default function App(){
  const [tab, setTab] = useState('soil')
  return (
    <div className="app">
      <header><h1>Soil & Vegetation Detection</h1>
        <div className="tabs">
          <button className={tab==='soil'?'active':''} onClick={()=>setTab('soil')}>Soil (YOLOv11)</button>
          <button className={tab==='veg'?'active':''} onClick={()=>setTab('veg')}>Vegetation (YOLOv8)</button>
        </div>
      </header>
      <main>
        {tab==='soil' ? <UploadPanel title="Soil Erosion Detection" endpoint="/predict/soil"/> : <UploadPanel title="Vegetation Detection" endpoint="/predict/vegetation"/>}
      </main>
      <footer>Server resizes images server-side to the model's target size and runs inference with conf=0.25</footer>
    </div>
  )
}

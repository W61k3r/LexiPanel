// === GG: Graph Gauntlet ======================================================
// A runner mini-game launched from any chart card's "▶ GG" button. The chart's
// own recent data becomes the opening stretch of track; after that the course
// is generated. Draw straight lines with the mouse/finger to bridge gaps, build
// ramps, and block arrows. Self-contained: no libraries, no network, no assets.
//
// Rules this code keeps (see docs/GG-AI-GUIDE.md before changing it):
//  * never render INSIDE a chart element - refresh() rewrites those every 5 s;
//    the game lives in its own .gghost next to them and the card hides the rest
//  * the animation loop stops completely whenever the game is paused, over,
//    closed, on a hidden tab or in a hidden browser tab (zero CPU when idle)
//  * the engine (makeWorld/step) is pure logic with no DOM, so it is testable
// ============================================================================
(function(){
'use strict';

const T={
  VIEW_H:12,          // metres of world visible top to bottom
  RUNNER_AT:0.3,      // runner's place across the view (0 = left edge)
  G:26,               // gravity on the runner, m/s²
  ARROW_G:9.8,        // gravity on arrows
  SPEED0:6, SPEED_MAX:14, SPEED_PER_M:0.02,   // speed grows with distance
  LINES0:3, LINES_MAX:6, LEN0:4.2, LEN_MAX:9, LEN_STEP:1.0, MIN_LINE:0.5,
  RECYCLE_M:20,       // a line comes back once it is this far behind the runner
  STEP:0.35,          // how far a surface may sit above/below the feet and still be "the floor"
  RUN_SLOPE:1.7,      // lines steeper than ~60° cannot be run on (you pass through them)
  FALL_Y:-7,          // below this the runner has fallen out of the graph
  DT:1/120,
};

// ---- small helpers ---------------------------------------------------------
function rng(seed){ let s=(seed>>>0)||1; return ()=>((s=Math.imul(s^s>>>15,1|s)+0x6D2B79F5|0,
  (((s^s>>>7)>>>0)%1e9)/1e9)); }
const lerp=(a,b,t)=>a+(b-a)*t;
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function yAt(s,x){ return s.x2===s.x1 ? Math.max(s.y1,s.y2) : s.y1+(s.y2-s.y1)*(x-s.x1)/(s.x2-s.x1); }
function slope(s){ return s.x2===s.x1 ? Infinity : (s.y2-s.y1)/(s.x2-s.x1); }
function segHit(ax,ay,bx,by,cx,cy,dx,dy){           // do segments AB and CD cross?
  const d=(bx-ax)*(dy-cy)-(by-ay)*(dx-cx); if(!d) return false;
  const u=((cx-ax)*(dy-cy)-(cy-ay)*(dx-cx))/d, v=((cx-ax)*(by-ay)-(cy-ay)*(bx-ax))/d;
  return u>=0&&u<=1&&v>=0&&v<=1;
}
function distToSeg(px,py,ax,ay,bx,by){
  const vx=bx-ax, vy=by-ay, l=vx*vx+vy*vy, t=l?clamp(((px-ax)*vx+(py-ay)*vy)/l,0,1):0;
  return Math.hypot(px-ax-t*vx, py-ay-t*vy);
}

// ---- the world ---------------------------------------------------------------
// series: numbers from the chart the game was launched from (may be empty)
function makeWorld(seed, series){
  const w={t:0, rnd:rng(seed), ground:[], drawn:[], spikes:[], archers:[], arrows:[], pickups:[],
    fx:[], genX:0, genY:0, lastPickupX:0,
    r:{x:2, y:0, vx:T.SPEED0, vy:0, on:null, phase:0},
    startX:2, dist:0, lines:T.LINES0, owned:T.LINES0, maxLen:T.LEN0,
    over:null, events:[]};
  addGround(w, 0, 0, 6, 0);                          // a flat start
  seedFromSeries(w, series);
  addGround(w, w.genX, w.genY, w.genX+8, w.genY);     // breathing room before hazards
  generate(w, 90);
  w.r.on=w.ground[0];
  return w;
}

function addGround(w,x1,y1,x2,y2){
  const s={x1,y1,x2,y2,kind:'ground'}; w.ground.push(s); w.genX=x2; w.genY=y2; return s;
}

// The chart's last points become 30 m of track, scaled so no slope is steeper
// than ~30° and the whole stretch spans at most 3 m of height.
function seedFromSeries(w, series){
  const d=(series||[]).filter(v=>typeof v==='number'&&isFinite(v)).slice(-40);
  if(d.length<3){ return; }
  const lo=Math.min(...d), hi=Math.max(...d), span=(hi-lo)||1, n=d.length, len=30, dx=len/(n-1);
  let amp=3;
  for(let i=1;i<n;i++){ const sl=Math.abs(d[i]-d[i-1])/span*amp/dx; if(sl>0.55) amp*=0.55/sl; }
  const x0=w.genX, y0=w.genY, base=(d[0]-lo)/span*amp;
  for(let i=1;i<n;i++){
    addGround(w, x0+(i-1)*dx, y0+(d[i-1]-lo)/span*amp-base, x0+i*dx, y0+(d[i]-lo)/span*amp-base);
  }
  w.seeded=true;
}

// Course pieces, harder with distance: rolling ground, pits, spike beds,
// archers, pickups. Heights stay within a band so the view never loses the track.
function generate(w, untilX){
  while(w.genX<untilX){
    const R=w.rnd, x=w.genX, y=w.genY, diff=clamp((x-40)/600,0,1);
    const early=x<45;
    const pick=early?0:R();
    if(pick<0.30){                                           // rolling ground
      let cx=x, cy=y;
      for(let k=0;k<2+(R()*3|0);k++){
        const l=3+R()*5; let ny=cy+(R()-0.5)*l*0.5; ny=clamp(ny,-2,4);
        addGround(w,cx,cy,cx+l,ny); cx+=l; cy=ny;
      }
    }else if(pick<0.55){                                     // pit
      const gap=2.4+diff*3.4+R()*1.4, ny=clamp(y+(R()-0.5)*1.6,-2,4);
      w.genX=x+gap; w.genY=ny;
      addGround(w,w.genX,ny,w.genX+5,ny);
    }else if(pick<0.72){                                     // spike bed
      addGround(w,x,y,x+3,y);
      const wdt=1.0+diff*2.0+R()*0.8;
      w.spikes.push({x1:x+3, x2:x+3+wdt, y});
      addGround(w,x+3,y,x+3+wdt,y); addGround(w,x+3+wdt,y,x+7+wdt,y);
    }else if(pick<0.86){                                     // archer
      addGround(w,x,y,x+14,y);
      w.archers.push({x:x+12, y, next:0, dead:false, fell:0, quiver:diff<0.25?1:(diff<0.6?2:3), aim:0});
    }else{                                                   // pit with spikes after it
      const gap=2.2+diff*2.2+R();
      w.genX=x+gap; addGround(w,w.genX,y,w.genX+2,y);
      const wdt=0.8+diff*1.2; w.spikes.push({x1:w.genX, x2:w.genX+wdt, y});
      addGround(w,w.genX,y,w.genX+wdt,y); addGround(w,w.genX,y,w.genX+4,y);
    }
    // pickups: more +LINE while lines are scarce, some low (run through), some high (need a ramp)
    if(w.genX-w.lastPickupX>16 && R()<0.55){
      const gs=w.ground[w.ground.length-1], px=(gs.x1+gs.x2)/2;
      const high=R()<0.45+diff*0.3;
      w.pickups.push({x:px, y:yAt(gs,px)+(high?2.6+R()*0.8:1.0), kind:R()<0.6?'line':'len', got:false});
      w.lastPickupX=w.genX;
    }
  }
}

function surfaces(w){ return w.ground.concat(w.drawn); }

// highest runnable surface under x whose height is within `below`..`above` of y
function floorAt(w, x, y, below, above, except){
  let best=null, by=-Infinity;
  for(const s of surfaces(w)){
    if(s===except || x<s.x1-1e-6 || x>s.x2+1e-6 || Math.abs(slope(s))>T.RUN_SLOPE) continue;
    const sy=yAt(s,x);
    if(sy>=y-below && sy<=y+above && sy>by){ best=s; by=sy; }
  }
  return best;
}

function die(w, why){ if(!w.over){ w.over={why, score:Math.floor(w.dist)}; w.events.push('die'); } }

function step(w, dt){
  if(w.over) return;
  const r=w.r; w.t+=dt;
  r.vx=Math.min(T.SPEED_MAX, T.SPEED0+w.dist*T.SPEED_PER_M);
  const nx=r.x+r.vx*dt;
  if(r.on){                                               // running on something
    const f=floorAt(w, nx, r.y, T.STEP, T.STEP, null);
    if(f){ if(f!==r.on && f.kind==='drawn') w.events.push('ramp'); r.on=f; r.x=nx; r.y=yAt(f,nx); r.vy=0; }
    else{                                                 // ran off the end: launch along the slope
      r.vy=r.vx*clamp(slope(r.on),-2,T.RUN_SLOPE); r.on=null; r.x=nx; r.y+=r.vy*dt;
      if(r.vy>1) w.events.push('launch');
    }
  }else{                                                  // in the air
    const py=r.y; r.vy-=T.G*dt; const ny=r.y+r.vy*dt;
    let land=null, ly=-Infinity;
    if(r.vy<=0){
      for(const s of surfaces(w)){
        if(nx<s.x1||nx>s.x2||Math.abs(slope(s))>T.RUN_SLOPE) continue;
        const sPrev=yAt(s,clamp(r.x,s.x1,s.x2)), sNow=yAt(s,nx);
        if(py>=sPrev-0.05 && ny<=sNow && sNow>ly){ land=s; ly=sNow; }
      }
    }
    r.x=nx;
    if(land){ r.on=land; r.y=ly; r.vy=0; w.events.push('land'); } else r.y=ny;
  }
  r.phase+=r.vx*dt*(r.on?1.9:0.6);
  w.dist=r.x-w.startX;

  if(r.y<T.FALL_Y) return die(w,'Your runner fell off the graph.');
  for(const s of w.spikes){
    if(r.x>s.x1-0.15 && r.x<s.x2+0.15 && r.y<s.y+0.45) return die(w,'Your runner ran onto the spikes.');
  }
  for(const p of w.pickups){
    if(p.got || Math.hypot(p.x-r.x, p.y-(r.y+0.9))>1.05) continue;
    p.got=true; w.events.push('pickup');
    if(p.kind==='line'){ if(w.owned<T.LINES_MAX){ w.owned++; w.lines++; } }
    else w.maxLen=Math.min(T.LEN_MAX,w.maxLen+T.LEN_STEP);
    w.fx.push({x:p.x,y:p.y,t:0,text:p.kind==='line'?'+ LINE':'+ LENGTH',kind:p.kind});
  }
  for(const l of w.drawn){                                  // lines far behind come back
    if(!l.back && l.x2<r.x-T.RECYCLE_M && l!==r.on){ l.back=true; w.lines=Math.min(w.owned,w.lines+1); w.events.push('back'); }
  }

  const diff=clamp(w.dist/600,0,1);
  for(const a of w.archers){
    if(!a.passed && r.x>a.x+1){ a.passed=true; if(!a.dead) w.events.push('pass'); }   // avoided, not tackled
    if(a.dead){ a.fell=Math.min(1,a.fell+dt*4); continue; }
    if(Math.abs(r.x-a.x)<0.45 && Math.abs(r.y-a.y)<1.2){ a.dead=true; w.events.push('tackle'); continue; }
    const ahead=a.x-r.x;
    a.aim=Math.max(0,a.aim-dt);
    if(a.quiver>0 && ahead>6 && ahead<30 && !a.drawing && w.t>=a.next){
      a.drawing=true; a.aim=0.6; continue;                  // a visible draw before each shot
    }
    if(a.drawing && a.aim<=0){
      a.drawing=false;
      const ft=1.35+w.rnd()*0.4, ax=a.x-0.35, ay=a.y+1.35;       // slow, high arcs: time to react
      const miss=(w.rnd()-0.5)*lerp(3.0,1.2,diff);         // not every arrow is on target
      const tx=r.x+r.vx*ft, ty=r.y+1.0+miss;               // lead the runner
      w.arrows.push({x:ax,y:ay,vx:(tx-ax)/ft,vy:(ty-ay+0.5*T.ARROW_G*ft*ft)/ft,dead:false});
      a.quiver--; a.next=w.t+lerp(1.6,1.0,diff)+w.rnd()*0.5; w.events.push('shoot');
    }
  }
  for(const ar of w.arrows){
    if(ar.dead) continue;
    const ox=ar.x, oy=ar.y; ar.vy-=T.ARROW_G*dt; ar.x+=ar.vx*dt; ar.y+=ar.vy*dt;
    for(const l of w.drawn){
      if(segHit(ox,oy,ar.x,ar.y,l.x1,l.y1,l.x2,l.y2)){ ar.dead=true; w.events.push('block');
        w.fx.push({x:ar.x,y:ar.y,t:0,spark:true}); break; }
    }
    if(ar.dead) continue;
    const g=floorAt(w, ar.x, ar.y, 0, 50, null);
    if(ar.y<-8 || (g && g.kind==='ground' && yAt(g,ar.x)>ar.y)) { ar.dead=true; w.events.push('dodge'); continue; }
    if(distToSeg(ar.x,ar.y,r.x,r.y+0.25,r.x,r.y+1.55)<0.32) return die(w,"An archer's arrow found your runner.");
  }
  generate(w, r.x+70);
  const keep=r.x-30;                                        // forget what scrolled away
  if(w.ground.length>400) w.ground=w.ground.filter(s=>s.x2>keep||s===r.on);
  w.drawn=w.drawn.filter(s=>!s.back);
  w.spikes=w.spikes.filter(s=>s.x2>keep); w.pickups=w.pickups.filter(p=>p.x>keep&&!p.got);
  w.archers=w.archers.filter(a=>a.x>keep); w.arrows=w.arrows.filter(a=>!a.dead&&a.x>keep-10);
  for(const f of w.fx) f.t+=dt; w.fx=w.fx.filter(f=>f.t<1.2);
}

// commit a drawn line; returns false (and changes nothing) if not allowed
function drawLine(w, x1,y1,x2,y2){
  if(w.over || w.lines<=0) return false;
  let len=Math.hypot(x2-x1,y2-y1);
  if(len<T.MIN_LINE) return false;
  if(len>w.maxLen){ const k=w.maxLen/len; x2=x1+(x2-x1)*k; y2=y1+(y2-y1)*k; }
  if(x2<x1){ [x1,x2]=[x2,x1]; [y1,y2]=[y2,y1]; }
  w.drawn.push({x1,y1,x2,y2,kind:'drawn'}); w.lines--; w.events.push('draw');
  return true;
}

// ---- the view ------------------------------------------------------------------
let G=null;                                                // the one open game

function css(name, fb){ const v=getComputedStyle(document.documentElement).getPropertyValue(name).trim(); return v||fb; }
// ---- run stats, personal bests, achievements: kept in this browser, like the record ----
const STAT={pass:'archers passed', block:'arrows blocked', dodge:'arrows dodged', tackle:'archers tackled',
            pickup:'pickups', draw:'lines drawn', launch:'jumps'};
const SESSION={runs:0, m:0};                                // since this page was loaded
const ACH=[
  ['steps','First steps','run 100 m',s=>s.m>=100],
  ['half','Half a kilometre','500 m in one run',s=>s.m>=500],
  ['km','Kilometre club','1,000 m in one run',s=>s.m>=1000],
  ['hike','Session hiker','2,000 m in one session',(s,S)=>S.m>=2000],
  ['road','Road warrior','10,000 m in all',(s,S,L)=>L.m>=10000],
  ['avoid','Archer avoider','pass 15 archers in one run',s=>(s.pass||0)>=15],
  ['ghost','Untouchable','pass 40 archers in one run',s=>(s.pass||0)>=40],
  ['shield','Shield wall','block 10 arrows in one run',s=>(s.block||0)>=10],
  ['umbrella','Umbrella','block 100 arrows in all',(s,S,L)=>(L.block||0)>=100],
  ['dodger','Artful dodger','25 arrows miss you in one run',s=>(s.dodge||0)>=25],
  ['tackle','Linebacker','tackle 5 archers in one run',s=>(s.tackle||0)>=5],
  ['collect','Collector','10 pickups in one run',s=>(s.pickup||0)>=10],
  ['architect','Architect','draw 40 lines in one run',s=>(s.draw||0)>=40],
  ['flyer','Frequent flyer','20 jumps in one run',s=>(s.launch||0)>=20],
  ['persist','Persistent','10 runs in one session',(s,S)=>S.runs>=10],
];
function readJSON(k,d){ try{ return JSON.parse(localStorage.getItem(k))||d; }catch(e){ return d; } }
function saveJSON(k,v){ try{ localStorage.setItem(k,JSON.stringify(v)); }catch(e){} }
function tally(w, ev){ const s=w.stats||(w.stats={}); for(const e of ev) if(e in STAT) s[e]=(s[e]||0)+1; }
function finishRun(w){
  const s=Object.assign({m:Math.floor(w.dist)}, w.stats||{});
  const L=readJSON('lexipanel.gg.stats',{}), B=readJSON('lexipanel.gg.bests',{}), got=readJSON('lexipanel.gg.ach',{});
  SESSION.runs++; SESSION.m+=s.m; L.runs=(L.runs||0)+1;
  const newBest=[];
  for(const k of ['m',...Object.keys(STAT)]){
    const v=s[k]||0; L[k]=(L[k]||0)+v;
    if(v>0 && v>(B[k]||0)){ if(B[k]) newBest.push(k); B[k]=v; }
  }
  const unlocked=[];
  for(const [id,name,,test] of ACH) if(!got[id] && test(s,SESSION,L)){ got[id]=new Date().toISOString().slice(0,10); unlocked.push(name); }
  saveJSON('lexipanel.gg.stats',L); saveJSON('lexipanel.gg.bests',B); saveJSON('lexipanel.gg.ach',got);
  return {s, B, L, newBest, unlocked, got};
}

function readRecord(){ try{ return +localStorage.getItem('lexipanel.gg.record')||0; }catch(e){ return 0; } }
function saveRecord(v){ try{ localStorage.setItem('lexipanel.gg.record',String(v)); }catch(e){} }

function open(card, opt){
  if(G && G.card===card) return close();
  if(G) close();
  opt=opt||{};
  const host=document.createElement('div');
  host.className='gghost'; host.tabIndex=0;
  host.setAttribute('role','application'); host.setAttribute('aria-label','Graph Gauntlet game');
  host.innerHTML=`<canvas></canvas>
    <div class="ggtitle">GRAPH GAUNTLET<span>${esc(opt.label?'track: '+opt.label:'')}</span></div>
    <div class="ggscore"><div><b data-k="m">0000</b> <small>METRES</small></div><div data-k="v">6.0 m/s</div></div>
    <div class="ggctl">
      <button type="button" data-a="pause">Pause</button><button type="button" data-a="restart">Restart</button>
      <button type="button" data-a="sound">Sound: off</button><button type="button" data-a="close">Graph</button></div>
    <div class="ggstat"><div data-k="lines">Available lines: 2</div><div data-k="len">4.2 m per line</div></div>
    <div class="ggover" data-k="over"></div>`;
  const h2=card.querySelector('h2');
  h2.insertAdjacentElement('afterend',host);
  card.classList.add('ggon');
  G={card, host, opt, cv:host.querySelector('canvas'), raf:0, last:0, acc:0, state:'ready',
     world:null, drag:null, sound:false, ac:null, record:readRecord(), camY:0, dpr:1, W:0, H:0};
  G.ctx=G.cv.getContext('2d');
  if(opt.button){ opt.button.textContent='■ Graph'; opt.button.classList.add('on'); }
  host.addEventListener('click',onClick);
  host.addEventListener('keydown',onKey);
  G.cv.addEventListener('pointerdown',onDown);
  G.cv.addEventListener('pointermove',onMove);
  G.cv.addEventListener('pointerup',onUp);
  G.cv.addEventListener('pointercancel',()=>{ if(G) G.drag=null; });
  G.ro=typeof ResizeObserver==='function'?new ResizeObserver(()=>{ resize(); if(G&&!G.raf) render(); }):null;
  if(G.ro) G.ro.observe(host); else window.addEventListener('resize',resize);
  resize(); newGame(); overlay('ready'); render();
  host.focus({preventScroll:true});
}

function close(){
  if(!G) return;
  stop();
  if(G.ro) G.ro.disconnect(); else window.removeEventListener('resize',resize);
  if(G.ac) try{ G.ac.close(); }catch(e){}
  G.host.remove(); G.card.classList.remove('ggon');
  if(G.opt.button){ G.opt.button.textContent='▶ GG'; G.opt.button.classList.remove('on'); }
  G=null;
}

function newGame(){
  let series=[];
  try{ series=(typeof G.opt.series==='function'?G.opt.series():G.opt.series)||[]; }catch(e){}
  G.world=makeWorld((Math.random()*2**31)|0, series);
  G.camY=0; G.drag=null; hud();
}

function start(){ if(!G||G.raf) return; G.state='run'; overlay(null); G.last=0; G.acc=0; G.raf=requestAnimationFrame(frame); }
function stop(){ if(G&&G.raf){ cancelAnimationFrame(G.raf); G.raf=0; } }
function pause(){ if(!G||G.state!=='run') return; stop(); G.state='paused'; G.drag=null; overlay('paused'); render(); }

function frame(ts){
  if(!G) return;
  // stop outright when nobody can see it: other tab, collapsed section, hidden browser tab
  if(!G.host.isConnected || G.host.offsetParent===null || document.hidden){ G.raf=0; pause(); return; }
  const dt=G.last?Math.min(0.05,(ts-G.last)/1000):T.DT; G.last=ts; G.acc+=dt;
  while(G.acc>=T.DT){ step(G.world,T.DT); G.acc-=T.DT; }
  sounds();
  render(); hud();
  if(G.world.over){ G.raf=0; gameOver(); return; }
  G.raf=requestAnimationFrame(frame);
}

function gameOver(){
  G.state='over';
  const s=G.world.over.score, best=s>G.record;
  if(best){ G.record=s; saveRecord(s); }
  overlay('over',{score:s, record:G.record, best, why:G.world.over.why, run:finishRun(G.world)});
}

function overlay(kind, o){
  const el=G.host.querySelector('[data-k="over"]');
  if(!kind){ el.className='ggover'; el.innerHTML=''; return; }
  el.className='ggover on';
  if(kind==='ready') el.innerHTML=`<div class="ggbox"><h3>Graph Gauntlet</h3>
      <p>Your runner sprints along ${G.world.seeded?'this graph\'s own line':'the graph'} and keeps going.
      <b>Drag</b> to draw straight lines: bridges over gaps, ramps to jump spikes and reach pickups,
      and roofs overhead to catch the archers' arrows.</p>
      <p class="ggkeys">You can have only a few lines out at once; each comes back when it is ${T.RECYCLE_M} m behind you.
      <span class="gp">+ LINE</span> and <span class="gl">+ LENGTH</span> give you more and longer ones.<br>
      P pause · R restart · Esc back to the graph</p>
      <p class="ggkeys">Record: <b>${G.record}</b> m · achievements ${Object.keys(readJSON('lexipanel.gg.ach',{})).length}/${ACH.length}</p>
      <button type="button" data-a="start">Start</button> <button type="button" data-a="ach">Achievements</button></div>`;
  else if(kind==='paused') el.innerHTML=`<div class="ggbox"><h3>Paused</h3>
      <button type="button" data-a="resume">Resume</button> <button type="button" data-a="close">Back to graph</button></div>`;
  else if(kind==='ach'){ const got=readJSON('lexipanel.gg.ach',{}), L=readJSON('lexipanel.gg.stats',{});
    el.innerHTML=`<div class="ggbox"><h3>Achievements ${Object.keys(got).length}/${ACH.length}</h3>
      <div class="ggach">${ACH.map(([id,n,d])=>`<div class="${got[id]?'ggyes':'ggno'}">${got[id]?'★':'☆'} <b>${esc(n)}</b>: ${esc(d)}${got[id]?` <small>${got[id]}</small>`:''}</div>`).join('')}</div>
      <p class="ggkeys">In all: ${L.runs||0} run${(L.runs||0)===1?'':'s'}, ${L.m||0} m, ${L.pass||0} archers passed, ${L.block||0} arrows blocked, ${L.dodge||0} dodged</p>
      <button type="button" data-a="again">Play</button> <button type="button" data-a="close">Back to graph</button></div>`; }
  else { const R=o.run||{s:{},B:{},newBest:[],unlocked:[],got:{}}, lab=Object.assign({m:'metres'},STAT);
    const rows=['m','pass','block','dodge','tackle','pickup'].map(k=>`<div>${lab[k]}: <b>${R.s[k]||0}</b> <small>best ${R.B[k]||0}</small>${R.newBest.includes(k)?' <span class="gl">NEW BEST</span>':''}</div>`).join('');
    const next=ACH.filter(a=>!R.got[a[0]]).slice(0,2).map(a=>a[1]+': '+a[2]).join(' · ');
    el.innerHTML=`<div class="ggbox"><h3>${o.best&&o.score>0?'New record!':'Game over'}</h3>
      <p>Your score: <b>${o.score}</b> m<br>Record: <b>${o.record}</b> m</p><p class="ggwhy">${esc(o.why)}</p>
      <div class="ggstats">${rows}</div>
      ${R.unlocked.length?`<p class="gp">Achievement${R.unlocked.length>1?'s':''} unlocked: ${esc(R.unlocked.join(', '))}</p>`:''}
      <p class="ggkeys">This session: ${SESSION.runs} run${SESSION.runs===1?'':'s'}, ${SESSION.m} m · achievements ${Object.keys(R.got).length}/${ACH.length}${next?'<br>Next: '+esc(next):''}</p>
      <button type="button" data-a="again">Try again</button> <button type="button" data-a="ach">Achievements</button> <button type="button" data-a="close">Back to graph</button></div>`; }
  const b=el.querySelector('button'); if(b) b.focus({preventScroll:true});
}

function hud(){
  const w=G.world, q=k=>G.host.querySelector(`[data-k="${k}"]`);
  q('m').textContent=String(Math.floor(w.dist)).padStart(4,'0');
  q('v').textContent=w.r.vx.toFixed(1)+' m/s';
  const L=q('lines'); L.textContent='Available lines: '+w.lines; L.classList.toggle('empty',w.lines===0);
  q('len').textContent=w.maxLen.toFixed(1)+' m per line';
}

function onClick(e){
  const a=e.target.closest('[data-a]'); if(!a||!G) return;
  const act=a.dataset.a;
  if(act==='start'||act==='resume') start();
  else if(act==='pause') G.state==='run'?pause():(G.state==='paused'&&start());
  else if(act==='restart'||act==='again'){ stop(); newGame(); render(); start(); }
  else if(act==='sound'){ G.sound=!G.sound; a.textContent='Sound: '+(G.sound?'on':'off'); if(G.sound) beep(660,0.06); }
  else if(act==='ach') overlay('ach');
  else if(act==='close') close();
  if(G) G.host.focus({preventScroll:true});
}

function onKey(e){
  if(!G) return;
  const k=e.key.toLowerCase();
  if(k==='escape'){ e.preventDefault(); close(); }
  else if(k==='p'){ e.preventDefault(); G.state==='run'?pause():((G.state==='paused'||G.state==='ready')&&start()); }
  else if(k==='r'){ e.preventDefault(); stop(); newGame(); render(); start(); }
}

// ---- drawing lines with the pointer -----------------------------------------------
function toWorld(e){
  const b=G.cv.getBoundingClientRect(), s=G.H/T.VIEW_H;
  return {x:G.camX+(e.clientX-b.left)/s, y:G.camY+(G.H-(e.clientY-b.top))/s};
}
function onDown(e){
  if(!G||G.state!=='run'||(e.button!==undefined&&e.button!==0)) return;
  e.preventDefault();
  if(G.world.lines<=0){ flash(); return; }
  const p=toWorld(e); G.drag={x1:p.x,y1:p.y,sx:e.clientX,sy:e.clientY,ex:e.clientX,ey:e.clientY};
  try{ G.cv.setPointerCapture(e.pointerId); }catch(_){}
}
function onMove(e){ if(G&&G.drag){ G.drag.ex=e.clientX; G.drag.ey=e.clientY; } }
function onUp(e){
  if(!G||!G.drag) return;
  const d=G.drag; G.drag=null; const p=toWorld(e);
  if(!drawLine(G.world,d.x1,d.y1,p.x,p.y) && G.world.lines<=0) flash();
  hud();
}
function dragEnd(){                                       // where the preview line ends, in world metres
  const d=G.drag, p=toWorld({clientX:d.ex,clientY:d.ey});
  let dx=p.x-d.x1, dy=p.y-d.y1; const l=Math.hypot(dx,dy);
  if(l>G.world.maxLen){ dx*=G.world.maxLen/l; dy*=G.world.maxLen/l; }
  return {x:d.x1+dx, y:d.y1+dy, len:Math.min(l,G.world.maxLen)};
}
function flash(){ const L=G.host.querySelector('[data-k="lines"]'); L.classList.remove('flash'); void L.offsetWidth; L.classList.add('flash'); beep(180,0.12,'square'); }

// ---- sound (off until asked for; synthesized, no files) ------------------------------
function beep(f,d,type){
  if(!G||!G.sound) return;
  try{
    if(!G.ac){ const AC=window.AudioContext||window.webkitAudioContext; if(!AC) return; G.ac=new AC(); }
    const o=G.ac.createOscillator(), g=G.ac.createGain(), t=G.ac.currentTime;
    o.type=type||'triangle'; o.frequency.setValueAtTime(f,t);
    g.gain.setValueAtTime(0.08,t); g.gain.exponentialRampToValueAtTime(0.0001,t+d);
    o.connect(g).connect(G.ac.destination); o.start(t); o.stop(t+d+0.02);
  }catch(e){}
}
function sounds(){
  const ev=G.world.events; G.world.events=[];
  tally(G.world, ev);
  if(!G.sound) return;
  for(const e of new Set(ev)){
    if(e==='pickup') beep(880,0.12); else if(e==='draw') beep(520,0.05);
    else if(e==='launch') beep(400,0.08,'sine'); else if(e==='block') beep(240,0.05,'square');
    else if(e==='die') beep(110,0.4,'sawtooth'); else if(e==='tackle') beep(300,0.06);
  }
}

// ---- rendering -------------------------------------------------------------------------
function resize(){
  if(!G) return;
  const b=G.host.getBoundingClientRect(), dpr=Math.max(1,Math.min(3,window.devicePixelRatio||1));
  G.dpr=dpr; G.W=Math.max(1,b.width); G.H=Math.max(1,b.height);
  G.cv.width=Math.round(G.W*dpr); G.cv.height=Math.round(G.H*dpr);
  G.cv.style.width=G.W+'px'; G.cv.style.height=G.H+'px';
}

function render(){
  if(!G||!G.ctx) return;
  const c=G.ctx, w=G.world, r=w.r, s=G.H/T.VIEW_H, viewW=G.W/s;
  const col={bg:css('--bg','#0f1115'), line:css('--line','#262b36'), fg:css('--fg','#e6e9ef'),
    dim:css('--dim','#9aa4b8'), track:css('--series-1','#3987e5'), fail:css('--fail','#f85149'),
    warn:css('--warn','#d29922'), gold:'#e3b341'};
  G.camX=r.x-viewW*T.RUNNER_AT;
  G.camY=lerp(G.camY, r.y-T.VIEW_H*0.38, G.state==='run'?0.08:1);
  const X=x=>(x-G.camX)*s, Y=y=>G.H-(y-G.camY)*s;
  c.setTransform(G.dpr,0,0,G.dpr,0,0);
  c.fillStyle=col.bg; c.fillRect(0,0,G.W,G.H);

  // graph paper: a gridline every 5 m across, 2 m up, labelled like the charts
  c.strokeStyle=col.line; c.lineWidth=1; c.fillStyle=col.dim; c.font='10px ui-monospace,monospace';
  for(let gx=Math.floor(G.camX/5)*5; gx<G.camX+viewW; gx+=5){
    c.beginPath(); c.moveTo(X(gx)+.5,0); c.lineTo(X(gx)+.5,G.H); c.stroke();
    if(gx>=w.startX) c.fillText(Math.round(gx-w.startX)+' m',X(gx)+3,G.H-4);
  }
  for(let gy=Math.floor(G.camY/2)*2; gy<G.camY+T.VIEW_H; gy+=2){
    c.beginPath(); c.moveTo(0,Y(gy)+.5); c.lineTo(G.W,Y(gy)+.5); c.stroke();
  }

  // the track, drawn the way lineChart draws a series: soft fill under a 2.5px line
  const vis=w.ground.filter(g=>g.x2>G.camX-1&&g.x1<G.camX+viewW+1);
  c.fillStyle=col.track; c.globalAlpha=0.12;
  for(const g of vis){ c.beginPath(); c.moveTo(X(g.x1),Y(g.y1)); c.lineTo(X(g.x2),Y(g.y2));
    c.lineTo(X(g.x2),G.H); c.lineTo(X(g.x1),G.H); c.closePath(); c.fill(); }
  c.globalAlpha=1; c.strokeStyle=col.track; c.lineWidth=2.5; c.lineJoin='round'; c.lineCap='round';
  c.beginPath(); let px=null, py=null;
  for(const g of vis){
    if(px===null||Math.abs(g.x1-px)>1e-6||Math.abs(g.y1-py)>1e-6) c.moveTo(X(g.x1),Y(g.y1));
    c.lineTo(X(g.x2),Y(g.y2)); px=g.x2; py=g.y2;
  }
  c.stroke();

  // spikes
  c.fillStyle=col.fail;
  for(const sp of w.spikes){
    const n=Math.max(2,Math.round((sp.x2-sp.x1)/0.45)), wd=(sp.x2-sp.x1)/n;
    for(let i=0;i<n;i++){ const a=sp.x1+i*wd; c.beginPath(); c.moveTo(X(a),Y(sp.y)); c.lineTo(X(a+wd/2),Y(sp.y+0.55));
      c.lineTo(X(a+wd),Y(sp.y)); c.closePath(); c.fill(); }
  }

  // pickups, bobbing
  c.textAlign='center'; c.font='bold 10px ui-monospace,monospace';
  for(const p of w.pickups){
    const by=p.y+Math.sin(w.t*3+p.x)*0.12, colr=p.kind==='line'?col.fail:col.gold;
    c.fillStyle=colr; c.beginPath(); c.arc(X(p.x),Y(by),0.32*s,0,7); c.fill();
    c.fillStyle=col.bg; c.fillText('+',X(p.x),Y(by)+3.5);
    c.fillStyle=colr; c.fillText(p.kind==='line'?'+ LINE':'+ LENGTH',X(p.x),Y(by-0.62));
  }
  c.textAlign='left';

  // archers (orange) and arrows
  for(const a of w.archers) stick(c,X,Y,s,a.x,a.y,col.warn,{archer:true,fell:a.fell,aim:a.aim>0,face:-1});
  c.strokeStyle=col.fail; c.lineWidth=2;
  for(const ar of w.arrows){
    const l=Math.hypot(ar.vx,ar.vy)||1, ux=ar.vx/l*0.55, uy=ar.vy/l*0.55;
    c.beginPath(); c.moveTo(X(ar.x-ux),Y(ar.y-uy)); c.lineTo(X(ar.x),Y(ar.y)); c.stroke();
  }

  // the player's lines, and the one being drawn
  c.strokeStyle=col.fg; c.lineWidth=3;
  for(const l of w.drawn){ c.beginPath(); c.moveTo(X(l.x1),Y(l.y1)); c.lineTo(X(l.x2),Y(l.y2)); c.stroke(); }
  if(G.drag){
    const e=dragEnd(), ok=e.len>=T.MIN_LINE;
    c.setLineDash([6,5]); c.strokeStyle=ok?col.fg:col.fail; c.globalAlpha=0.8;
    c.beginPath(); c.moveTo(X(G.drag.x1),Y(G.drag.y1)); c.lineTo(X(e.x),Y(e.y)); c.stroke();
    c.setLineDash([]); c.globalAlpha=1;
    c.fillStyle=col.dim; c.fillText(e.len.toFixed(1)+' / '+w.maxLen.toFixed(1)+' m',X(e.x)+8,Y(e.y)-6);
  }

  // the runner
  if(!w.over || Math.floor(w.t*10)%2) stick(c,X,Y,s,r.x,r.y,col.fg,{phase:r.phase,air:!r.on,face:1});

  // floating texts and sparks
  for(const f of w.fx){
    c.globalAlpha=1-f.t/1.2;
    if(f.spark){ c.fillStyle=col.fg; for(let i=0;i<5;i++){ const a=i*1.26+f.t*4; c.fillRect(X(f.x+Math.cos(a)*f.t*1.4),Y(f.y+Math.sin(a)*f.t*1.4),2,2); } }
    else{ c.fillStyle=f.kind==='line'?col.fail:col.gold; c.font='bold 12px ui-monospace,monospace';
      c.fillText(f.text,X(f.x)-24,Y(f.y+f.t*1.5)); }
    c.globalAlpha=1;
  }
}

// a stick figure standing at feet (fx,fy), 1.6 m tall
function stick(c,X,Y,s,fx,fy,colr,o){
  c.save(); c.strokeStyle=colr; c.fillStyle=colr; c.lineWidth=Math.max(2,0.09*s); c.lineCap='round';
  if(o.archer&&o.fell){                                    // knocked over: lying down
    c.translate(X(fx),Y(fy)); c.rotate(-o.fell*Math.PI/2); c.translate(-X(fx),-Y(fy));
  }
  const hip={x:fx,y:fy+0.75}, neck={x:fx+(o.air?0.05:0.08),y:fy+1.25}, head={x:neck.x+0.02,y:fy+1.48};
  const ph=o.phase||0, sw=o.air?0.5:Math.sin(ph)*0.5, sw2=o.air?-0.3:Math.sin(ph+Math.PI)*0.5;
  const line=(a,b)=>{ c.beginPath(); c.moveTo(X(a.x),Y(a.y)); c.lineTo(X(b.x),Y(b.y)); c.stroke(); };
  line(hip,neck);
  c.beginPath(); c.arc(X(head.x),Y(head.y),0.2*s,0,7); c.lineWidth=Math.max(2,0.07*s); c.stroke(); c.lineWidth=Math.max(2,0.09*s);
  if(o.archer){
    const hand={x:fx-0.45,y:fy+1.2};
    line(neck,hand); line(neck,{x:fx-0.2,y:fy+0.95});
    c.beginPath(); c.arc(X(hand.x+0.12),Y(hand.y),0.42*s,Math.PI*0.62,Math.PI*1.38); c.stroke();   // the bow
    if(o.aim){ c.lineWidth=1; c.beginPath(); c.moveTo(X(hand.x+0.1),Y(hand.y+0.4)); c.lineTo(X(hand.x+0.3),Y(hand.y));
      c.lineTo(X(hand.x+0.1),Y(hand.y-0.4)); c.stroke(); }                                        // string drawn back
    line(hip,{x:fx-0.18,y:fy}); line(hip,{x:fx+0.18,y:fy});
  }else{
    line(neck,{x:neck.x+sw2*0.6,y:fy+0.85}); line(neck,{x:neck.x+sw*0.6,y:fy+0.85});
    const k1={x:hip.x+sw*0.35,y:fy+0.38}, k2={x:hip.x+sw2*0.35,y:fy+0.38};
    line(hip,k1); line(k1,{x:k1.x+sw*0.15-0.05,y:fy+(sw>0?0.08:0)});
    line(hip,k2); line(k2,{x:k2.x+sw2*0.15-0.05,y:fy+(sw2>0?0.08:0)});
  }
  c.restore();
}

function esc(s){ return String(s??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch])); }

document.addEventListener('visibilitychange',()=>{ if(document.hidden) pause(); });

window.GG={ open, close, toggle:open, isOpen:()=>!!G,
  _engine:{T, makeWorld, step, drawLine, floorAt, yAt} };        // for tests only
})();

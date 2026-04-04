/** Auth module — login, register, token management. */

const TOKEN_KEY = 'tq-auth-token';
const USER_KEY  = 'tq-auth-user';

let _token = localStorage.getItem(TOKEN_KEY);
let _user  = (() => {
  try { return JSON.parse(localStorage.getItem(USER_KEY)); } catch { return null; }
})();

export function getToken()    { return _token; }
export function getUser()     { return _user; }
export function isLoggedIn()  { return !!_token && !!_user && !_isTokenExpired(); }

/** Check if stored JWT has expired (client-side, no server call). */
function _isTokenExpired() {
  if (!_token) return true;
  try {
    const payload = JSON.parse(atob(_token.split('.')[1]));
    // Expired if exp is in the past (with 30s buffer)
    return payload.exp && payload.exp < (Date.now() / 1000) - 30;
  } catch {
    return true; // Malformed token
  }
}

function _save(token, user) {
  _token = token;
  _user  = user;
  localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, JSON.stringify(user));
}

export function logout() {
  _token = null;
  _user  = null;
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

/** Attach Authorization header to fetch calls. */
export function authHeaders() {
  return _token ? { 'Authorization': `Bearer ${_token}` } : {};
}

export async function apiRegister(username, password) {
  const r = await fetch('/v1/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || 'Registration failed');
  }
  const data = await r.json();
  _save(data.token, { user_id: data.user_id, username: data.username });
  return data;
}

export async function apiLogin(username, password) {
  const r = await fetch('/v1/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({ detail: r.statusText }));
    throw new Error(err.detail || 'Login failed');
  }
  const data = await r.json();
  _save(data.token, { user_id: data.user_id, username: data.username });
  return data;
}

export async function apiMe() {
  if (!_token) return null;
  const r = await fetch('/v1/auth/me', {
    headers: { ...authHeaders() },
  });
  if (!r.ok) { logout(); return null; }
  return r.json();
}

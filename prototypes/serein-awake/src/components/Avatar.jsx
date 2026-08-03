import { PencilSimple } from "@phosphor-icons/react";

export function Avatar({ person, src, size = "large", onReplace, editable = false }) {
  return (
    <div className={`avatar avatar--${size}`}>
      <img src={src} alt={`${person.name} 的头像`} style={{ objectPosition: person.position }} />
      {editable ? (
        <label className="avatar__edit" aria-label={`更换${person.name}的头像`}>
          <PencilSimple size={15} weight="light" aria-hidden="true" />
          <input type="file" accept="image/*" onChange={(event) => onReplace(event.target.files?.[0])} />
        </label>
      ) : null}
    </div>
  );
}
